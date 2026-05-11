#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Humble 在线层级导航运行时节点（第一版保守原型）。

设计目标：
1) 优先保证可运行与安全停车；
2) 图像 latest-only 缓存，避免回调积压；
3) 状态条件调度（state_conditioned_v2）+ 模型输出缓存复用；
4) 持续发布 cmd_vel / state / debug。
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
import yaml

try:
    from cv_bridge import CvBridge  # type: ignore
except Exception:  # pragma: no cover - 运行环境无 cv_bridge 时走 fallback
    CvBridge = None  # type: ignore


@dataclass
class ModuleCache:
    """保存模块缓存、异步任务状态与最近调度信息。"""

    last_output: Dict[str, Any]
    last_update_time: Optional[Time] = None
    busy: bool = False
    future: Optional[Future] = None
    last_run_step: int = 0
    last_latency_ms: Optional[float] = None
    last_start_wall_time: Optional[float] = None
    last_finish_wall_time: Optional[float] = None
    exception_count: int = 0
    success_count: int = 0


class HierarchicalNavRuntimeNode(Node):
    """真实小车层级导航在线运行时节点。"""

    def __init__(self) -> None:
        super().__init__("hierarchical_nav_runtime")
        self.runtime_build_tag = "v1h_recenter_debug_2026_04"
        self.runtime_file = os.path.abspath(__file__)

        # ===== 1) 读取配置路径参数并加载 YAML =====
        self.repo_root: Optional[str] = self._detect_repo_root(config_path=None)
        self.config_dir: Optional[str] = None
        self.declare_parameter("config_path", "configs/hierarchical_nav_robot_v1.yaml")
        cfg_path_raw = (
            self.get_parameter("config_path").get_parameter_value().string_value.strip()
        )
        self.config_path = self._resolve_config_path(cfg_path_raw)
        self.config_dir = str(Path(self.config_path).resolve().parent)
        self.config = self._load_config(self.config_path)

        # 支持 launch 覆盖 topic：非空字符串生效，空字符串回退到 yaml/default
        self.declare_parameter("image_topic", "")
        self.declare_parameter("cmd_vel_topic", "")
        self.declare_parameter("state_topic", "")
        self.declare_parameter("debug_topic", "")

        self.system_cfg = self._cfg_dict("system")
        self.models_cfg = self._cfg_dict("models")
        self.state_machine_cfg = self._cfg_dict("state_machine")
        self.turn_control_cfg = self._cfg_dict("turn_control")
        self.straight_keep_cfg = self._cfg_dict("straight_keep")
        self.scheduler_cfg = self._cfg_dict("scheduler")
        self.robot_control_cfg = self._cfg_dict("robot_control")
        self.safety_cfg = self._cfg_dict("safety")
        self.topics_cfg = self._cfg_dict("topics")

        # ===== 2) 按配置准备运行参数 =====
        ros_image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value.strip()
        )
        ros_cmd_vel_topic = (
            self.get_parameter("cmd_vel_topic").get_parameter_value().string_value.strip()
        )
        ros_state_topic = (
            self.get_parameter("state_topic").get_parameter_value().string_value.strip()
        )
        ros_debug_topic = (
            self.get_parameter("debug_topic").get_parameter_value().string_value.strip()
        )

        yaml_image_topic = str(self.topics_cfg.get("image_topic", "")).strip()
        yaml_cmd_vel_topic = str(self.topics_cfg.get("cmd_vel_topic", "")).strip()
        yaml_state_topic = str(self.topics_cfg.get("state_topic", "")).strip()
        yaml_debug_topic = str(self.topics_cfg.get("debug_topic", "")).strip()

        self.image_topic, image_topic_src = self._resolve_topic_value(
            launch_value=ros_image_topic,
            yaml_value=yaml_image_topic,
            default_value="/camera/image_raw",
        )
        self.cmd_vel_topic, cmd_vel_topic_src = self._resolve_topic_value(
            launch_value=ros_cmd_vel_topic,
            yaml_value=yaml_cmd_vel_topic,
            default_value="/cmd_vel",
        )
        self.state_topic, state_topic_src = self._resolve_topic_value(
            launch_value=ros_state_topic,
            yaml_value=yaml_state_topic,
            default_value="/nav/state",
        )
        self.debug_topic, debug_topic_src = self._resolve_topic_value(
            launch_value=ros_debug_topic,
            yaml_value=yaml_debug_topic,
            default_value="/nav/debug",
        )
        self._topic_source: Dict[str, str] = {
            "image_topic": image_topic_src,
            "cmd_vel_topic": cmd_vel_topic_src,
            "state_topic": state_topic_src,
            "debug_topic": debug_topic_src,
        }

        self.cmd_publish_hz = max(
            1.0, float(self.robot_control_cfg.get("cmd_publish_hz", 10.0))
        )
        self.angular_clip = max(
            0.0, float(self.robot_control_cfg.get("angular_clip", 0.25))
        )
        self.straight_keep_bias = float(
            self.robot_control_cfg.get("straight_keep_bias", 0.0)
        )
        self.straight_keep_scale = float(
            self.robot_control_cfg.get("straight_keep_scale", 1.0)
        )
        self.straight_keep_deadband = max(
            0.0, float(self.robot_control_cfg.get("straight_keep_deadband", 0.0))
        )
        self.max_omega_hold_sec = max(
            0.0, float(self.robot_control_cfg.get("max_omega_hold_sec", 0.4))
        )
        self.omega_stale_decay = self._clip(
            float(self.robot_control_cfg.get("omega_stale_decay", 0.5)), 0.0, 1.0
        )
        self.zero_omega_on_stale_inference = bool(
            self.robot_control_cfg.get("zero_omega_on_stale_inference", True)
        )
        self.omega_rate_limit_per_step = max(
            0.0, float(self.robot_control_cfg.get("omega_rate_limit_per_step", 0.06))
        )
        self.allow_linear_hold_on_stale_straight = bool(
            self.robot_control_cfg.get("allow_linear_hold_on_stale_straight", True)
        )
        self.linear_hold_speed_on_stale = max(
            0.0, float(self.robot_control_cfg.get("linear_hold_speed_on_stale", 0.04))
        )
        self.max_linear_hold_sec = max(
            0.0, float(self.robot_control_cfg.get("max_linear_hold_sec", 1.2))
        )
        self.linear_hold_require_trigger_straight = bool(
            self.robot_control_cfg.get("linear_hold_require_trigger_straight", True)
        )
        self.linear_hold_block_stage3_turn = bool(
            self.robot_control_cfg.get("linear_hold_block_stage3_turn", True)
        )
        self.linear_hold_max_image_age_sec = max(
            0.0,
            float(self.robot_control_cfg.get("linear_hold_max_image_age_sec", 0.5)),
        )
        # slow_straight_only 专用：脉冲式纠偏，非纠偏 tick 强制保持零角速度。
        self.pulse_recenter_enable = bool(
            self.robot_control_cfg.get("pulse_recenter_enable", False)
        )
        self.pulse_recenter_enter_abs = max(
            0.0, float(self.robot_control_cfg.get("pulse_recenter_enter_abs", 0.055))
        )
        self.pulse_recenter_exit_abs = max(
            0.0, float(self.robot_control_cfg.get("pulse_recenter_exit_abs", 0.025))
        )
        self.pulse_recenter_enter_votes = max(
            1, int(self.robot_control_cfg.get("pulse_recenter_enter_votes", 2))
        )
        self.pulse_recenter_exit_votes = max(
            1, int(self.robot_control_cfg.get("pulse_recenter_exit_votes", 1))
        )
        self.pulse_recenter_max_steps = max(
            1, int(self.robot_control_cfg.get("pulse_recenter_max_steps", 3))
        )
        self.pulse_recenter_cooldown_steps = max(
            0, int(self.robot_control_cfg.get("pulse_recenter_cooldown_steps", 4))
        )
        self.pulse_recenter_omega = max(
            0.0, float(self.robot_control_cfg.get("pulse_recenter_omega", 0.025))
        )
        self.pulse_recenter_stop_on_sign_flip = bool(
            self.robot_control_cfg.get("pulse_recenter_stop_on_sign_flip", True)
        )

        self.linear_speed_map: Dict[str, float] = {
            "BOOT": float(self.robot_control_cfg.get("linear_speed_boot", 0.0)),
            "STRAIGHTKEEP": float(
                self.robot_control_cfg.get("linear_speed_straightkeep", 0.2)
            ),
            "APPROACH": float(self.robot_control_cfg.get("linear_speed_approach", 0.2)),
            "PROVISIONAL_TURN": float(
                self.robot_control_cfg.get("linear_speed_provisional_turn", 0.15)
            ),
            "TURN": float(self.robot_control_cfg.get("linear_speed_turn", 0.1)),
            "RECOVER": float(self.robot_control_cfg.get("linear_speed_recover", 0.15)),
        }

        self.turn_left_omega = float(self.turn_control_cfg.get("left_omega", 0.5))
        self.turn_right_omega = float(self.turn_control_cfg.get("right_omega", -0.5))
        self.disable_junction_after_lock = bool(
            self.scheduler_cfg.get("disable_junction_after_lock", True)
        )

        self.scheduler_policy = str(
            self.scheduler_cfg.get("policy", "state_conditioned_v2")
        ).strip()
        self.stage3_probe_stride = max(
            1, int(self.scheduler_cfg.get("stage3_probe_stride", 8))
        )
        self.straight_keep_stride = max(
            1, int(self.scheduler_cfg.get("straight_keep_stride", 1))
        )
        self.trigger_stride = max(
            1, int(self.scheduler_cfg.get("trigger_stride", 2))
        )
        self.junction_probe_stride = max(
            1, int(self.scheduler_cfg.get("junction_probe_stride", 1))
        )
        self.turn_stage3_stride = max(
            1, int(self.scheduler_cfg.get("turn_stage3_stride", 2))
        )
        self.recover_stage3_stride = max(
            1, int(self.scheduler_cfg.get("recover_stage3_stride", 2))
        )
        infer_max_workers_raw = self.scheduler_cfg.get("infer_max_workers", None)
        if infer_max_workers_raw is None:
            infer_max_workers_raw = self.system_cfg.get("infer_max_workers", 1)
        self.infer_max_workers = max(1, int(infer_max_workers_raw))
        self.runtime_mode = str(
            self.scheduler_cfg.get("runtime_mode", "event_v2")
        ).strip()
        self.single_active_model = bool(
            self.scheduler_cfg.get("single_active_model", False)
        )
        self.global_max_inflight_models = max(
            1,
            int(
                self.scheduler_cfg.get(
                    "global_max_inflight_models", self.infer_max_workers
                )
            ),
        )
        self.max_submits_per_tick = max(
            1, int(self.scheduler_cfg.get("max_submits_per_tick", 99))
        )

        self.image_timeout_sec = max(
            0.01, float(self.safety_cfg.get("image_timeout_sec", 1.0))
        )
        self.model_output_timeout_sec = max(
            0.01, float(self.safety_cfg.get("model_output_timeout_sec", 1.0))
        )
        self.startup_warmup_sec = float(self.safety_cfg.get("startup_warmup_sec", 2.5))
        self.publish_zero_on_timeout = bool(
            self.safety_cfg.get("publish_zero_on_timeout", True)
        )
        self.publish_zero_on_missing_image = bool(
            self.safety_cfg.get("publish_zero_on_missing_image", True)
        )
        self.max_consecutive_errors = max(
            1, int(self.safety_cfg.get("max_consecutive_errors", 5))
        )
        self.debug_compact = bool(self.safety_cfg.get("debug_compact", True))

        # ===== 3) 动态导入仓库内模块（兼容源码运行与安装运行） =====
        self._prepare_repo_import_path(self.config_path)
        self.Stage3Infer, self.JunctionLRInfer, self.StraightKeepInfer, self.ApproachTriggerInfer = (
            self._import_infer_classes()
        )
        self.HierarchicalNavigatorStateMachine = self._import_state_machine_class()

        # ===== 4) 初始化模型、状态机 =====
        self.state_machine = self._build_state_machine()
        self.models: Dict[str, Any] = {}
        self._load_models()

        # ===== 5) latest-only 图像缓存（仅缓存，不推理） =====
        self._img_lock = Lock()
        self._latest_image: Optional[np.ndarray] = None
        self._latest_image_time: Optional[Time] = None
        self._latest_image_header_time: Optional[Time] = None
        self._latest_image_receive_time: Optional[Time] = None
        self._latest_image_stamp: Optional[Time] = None
        self._image_rx_count = 0
        self._has_received_first_image = False
        self._first_image_logged = False
        self._last_missing_image_log_time: Optional[Time] = None
        self._missing_image_log_interval_sec = max(1.0, float(self.image_timeout_sec))
        self._bridge = CvBridge() if CvBridge is not None else None
        self._bridge_warned = False
        self._module_names: Tuple[str, ...] = (
            "stage3",
            "junction_lr",
            "straight_keep",
            "approach_trigger",
        )
        self._module_lock = Lock()
        self._infer_executor = ThreadPoolExecutor(
            max_workers=self.infer_max_workers,
            thread_name_prefix="hier_nav_infer",
        )
        self._image_cb_group = ReentrantCallbackGroup()
        self._control_cb_group = ReentrantCallbackGroup()
        self._executor_shutdown = False

        # ===== 6) 各模型缓存输出 =====
        self.module_cache: Dict[str, ModuleCache] = {
            "stage3": ModuleCache(last_output=self._default_output("stage3")),
            "junction_lr": ModuleCache(last_output=self._default_output("junction_lr")),
            "straight_keep": ModuleCache(
                last_output=self._default_output("straight_keep")
            ),
            "approach_trigger": ModuleCache(
                last_output=self._default_output("approach_trigger")
            ),
        }

        # ===== 7) ROS2 通信对象 =====
        image_qos = QoSProfile(
            history=qos_profile_sensor_data.history,
            depth=1,  # latest-only: 只缓存最新一帧
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
        )
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            image_qos,
            callback_group=self._image_cb_group,
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self._node_start_time = self.get_clock().now()
        self._tick_count = 0
        self._consecutive_errors = 0
        self._last_cmd = Twist()
        self._last_angular_z = 0.0
        self._last_angular_cmd_valid = False
        self._debug_last_nav_state: Optional[str] = None
        self._debug_state_step_count = 0
        self._debug_turn_step_count = 0
        self._debug_recover_step_count = 0
        self._debug_last_turn_dir: Optional[str] = None
        self._debug_last_turn_sign: Optional[int] = None
        self._active_scheduled_module_this_tick: Optional[str] = None
        self._pulse_recenter_state = "CENTER_HOLD"
        self._pulse_recenter_dir = 0
        self._pulse_recenter_step_count = 0
        self._pulse_recenter_enter_count = 0
        self._pulse_recenter_exit_count = 0
        self._pulse_recenter_cooldown_count = 0
        self.control_timer = self.create_timer(
            1.0 / self.cmd_publish_hz,
            self._control_tick,
            callback_group=self._control_cb_group,
        )

        if self.scheduler_policy not in (
            "state_conditioned_v2",
            "slow_safe_straight_only",
        ):
            self.get_logger().warn(
                "scheduler.policy=%s，当前节点仅实现 state_conditioned_v2，已按该策略运行。"
                % self.scheduler_policy
            )

        self.get_logger().info(
            "HierarchicalNavRuntimeNode started. runtime_build_tag=%s, runtime_file=%s, "
            "config_path=%s, image_topic=%s, cmd_vel_topic=%s, state_topic=%s, "
            "debug_topic=%s, hz=%.2f, infer_max_workers=%d, runtime_mode=%s, "
            "single_active_model=%s, global_max_inflight_models=%d, max_submits_per_tick=%d"
            % (
                self.runtime_build_tag,
                self.runtime_file,
                self.config_path,
                self.image_topic,
                self.cmd_vel_topic,
                self.state_topic,
                self.debug_topic,
                self.cmd_publish_hz,
                self.infer_max_workers,
                self.runtime_mode,
                str(self.single_active_model),
                self.global_max_inflight_models,
                self.max_submits_per_tick,
            )
        )
        self.get_logger().info(
            "Resolved topics => image_topic=%s (%s), cmd_vel_topic=%s (%s), state_topic=%s (%s), "
            "debug_topic=%s (%s)"
            % (
                self.image_topic,
                self._topic_source["image_topic"],
                self.cmd_vel_topic,
                self._topic_source["cmd_vel_topic"],
                self.state_topic,
                self._topic_source["state_topic"],
                self.debug_topic,
                self._topic_source["debug_topic"],
            )
        )

    @staticmethod
    def _sign_of(v: float, eps: float = 1e-9) -> int:
        if float(v) > float(eps):
            return 1
        if float(v) < -float(eps):
            return -1
        return 0

    @staticmethod
    def _turn_dir_to_sign(turn_dir: Optional[str]) -> Optional[int]:
        if turn_dir == "Left":
            return 1
        if turn_dir == "Right":
            return -1
        return None

    @staticmethod
    def _sign_to_turn_dir(sign: Optional[int]) -> Optional[str]:
        if sign is None:
            return None
        if int(sign) > 0:
            return "Left"
        if int(sign) < 0:
            return "Right"
        return None

    @staticmethod
    def _is_model_omega_state(state: str) -> bool:
        return state in ("STRAIGHTKEEP", "APPROACH", "PROVISIONAL_TURN", "RECOVER")

    def _stale_cache_age_threshold_ms(self) -> int:
        if self.cmd_publish_hz <= 1e-6:
            return 200
        return max(200, int((2.0 / float(self.cmd_publish_hz)) * 1000.0))

    def _get_latest_inference_age_ms(self, now: Time) -> int:
        with self._module_lock:
            update_times = [
                self.module_cache[module_name].last_update_time
                for module_name in self._module_names
            ]
        valid_times = [t for t in update_times if t is not None]
        if not valid_times:
            return -1
        latest_time = max(valid_times, key=lambda t: int(t.nanoseconds))
        return int(self._age_ms(latest_time, now))

    def _get_cmd_inference_age_ms(self, state: str, now: Time) -> int:
        if self._is_model_omega_state(state):
            with self._module_lock:
                last_update_time = self.module_cache["straight_keep"].last_update_time
            return int(self._age_ms(last_update_time, now))
        return self._get_latest_inference_age_ms(now)

    def _rate_limit_omega(self, target: float, previous: float) -> float:
        limit = float(self.omega_rate_limit_per_step)
        if limit <= 0.0:
            return float(target)
        delta = float(target) - float(previous)
        if delta > limit:
            return float(previous) + limit
        if delta < -limit:
            return float(previous) - limit
        return float(target)

    def _empty_stale_omega_diag(self) -> Dict[str, Any]:
        return {
            "stale_omega_suppressed": False,
            "stale_omega_before": None,
            "stale_omega_after": None,
            "max_omega_hold_sec": float(self.max_omega_hold_sec),
            "omega_stale_decay": float(self.omega_stale_decay),
            "stale_linear_hold_active": False,
            "stale_linear_hold_allowed": False,
            "stale_linear_hold_reason": "",
            "linear_hold_speed_on_stale": float(self.linear_hold_speed_on_stale),
            "max_linear_hold_sec": float(self.max_linear_hold_sec),
            "linear_hold_max_image_age_sec": float(self.linear_hold_max_image_age_sec),
        }

    def _apply_stale_omega_policy(
        self,
        *,
        now: Time,
        state: str,
        linear_x: float,
        angular_z: float,
        reason: str,
        image_age_ms: int,
        trigger_pred: str,
        stage3_pred: str,
        completed_modules: Optional[List[str]],
    ) -> Tuple[float, float, Optional[str], Dict[str, Any], bool, bool]:
        cmd_new, cmd_cached = self._compute_cmd_origin_flags(
            state=state,
            completed_modules=completed_modules,
            cmd_from_new_inference=None,
            cmd_from_cached_output=None,
        )
        stale_diag = self._empty_stale_omega_diag()
        if (not self._is_model_omega_state(state)) or cmd_new:
            return (
                float(linear_x),
                float(angular_z),
                None,
                stale_diag,
                bool(cmd_new),
                bool(cmd_cached),
            )

        latest_inference_age_ms = self._get_cmd_inference_age_ms(state, now)
        stale_diag["stale_omega_suppressed"] = True
        stale_diag["stale_omega_before"] = float(angular_z)

        last_angular_z = (
            float(self._last_angular_z) if self._last_angular_cmd_valid else 0.0
        )
        hold_exceeded = (
            int(latest_inference_age_ms) < 0
            or (float(latest_inference_age_ms) / 1000.0) > float(self.max_omega_hold_sec)
        )
        source_override: Optional[str] = None
        adjusted_linear_x = float(linear_x)

        if hold_exceeded:
            hold_allowed = False
            hold_reason = "not_straightkeep"
            straight_keep_age_sec = (
                float(latest_inference_age_ms) / 1000.0
                if int(latest_inference_age_ms) >= 0
                else None
            )
            image_age_sec = (
                float(image_age_ms) / 1000.0 if int(image_age_ms) >= 0 else None
            )

            if not self.allow_linear_hold_on_stale_straight:
                hold_reason = "disabled"
            elif state != "STRAIGHTKEEP":
                hold_reason = "state_not_straightkeep"
            elif str(reason or "") != "ok":
                hold_reason = "reason_not_ok"
            elif image_age_sec is None:
                hold_reason = "image_age_unknown"
            elif image_age_sec > float(self.linear_hold_max_image_age_sec):
                hold_reason = "image_too_old"
            elif straight_keep_age_sec is None:
                hold_reason = "straight_keep_age_unknown"
            elif straight_keep_age_sec > float(self.max_linear_hold_sec):
                hold_reason = "straight_keep_too_old"
            elif (
                self.linear_hold_require_trigger_straight
                and str(trigger_pred or "") != "Straight"
            ):
                hold_reason = "trigger_not_straight"
            elif self.linear_hold_block_stage3_turn and str(stage3_pred or "") == "Turn":
                hold_reason = "stage3_turn"
            else:
                hold_allowed = True
                hold_reason = "ok"

            stale_diag["stale_linear_hold_allowed"] = bool(hold_allowed)
            stale_diag["stale_linear_hold_reason"] = str(hold_reason)

            if state == "STRAIGHTKEEP":
                adjusted_omega = 0.0
                if hold_allowed:
                    adjusted_linear_x = min(
                        float(linear_x), float(self.linear_hold_speed_on_stale)
                    )
                    source_override = "stale_straight_keep_linear_hold"
                    stale_diag["stale_linear_hold_active"] = True
                else:
                    adjusted_linear_x = 0.0
                    source_override = (
                        "stale_cache_zero"
                        if cmd_cached and int(latest_inference_age_ms) >= 0
                        else "stale_straight_keep_zero"
                    )
            elif self.zero_omega_on_stale_inference:
                adjusted_linear_x = 0.0
                adjusted_omega = 0.0
                source_override = (
                    "stale_cache_zero"
                    if cmd_cached and int(latest_inference_age_ms) >= 0
                    else "stale_omega_zero"
                )
            else:
                adjusted_linear_x = 0.0
                target_omega = last_angular_z * float(self.omega_stale_decay)
                clipped_omega = self._clip(
                    target_omega, -self.angular_clip, self.angular_clip
                )
                adjusted_omega = self._rate_limit_omega(clipped_omega, last_angular_z)
                source_override = "stale_cache_decay" if cmd_cached else "stale_omega_decay"
        else:
            target_omega = last_angular_z * float(self.omega_stale_decay)
            clipped_omega = self._clip(target_omega, -self.angular_clip, self.angular_clip)
            adjusted_omega = self._rate_limit_omega(clipped_omega, last_angular_z)
            source_override = "stale_cache_decay" if cmd_cached else "stale_omega_decay"

        if abs(adjusted_omega) < 1e-9:
            adjusted_omega = 0.0
        stale_diag["stale_omega_after"] = float(adjusted_omega)
        return (
            float(adjusted_linear_x),
            float(adjusted_omega),
            source_override,
            stale_diag,
            bool(cmd_new),
            bool(cmd_cached),
        )

    def _compute_cmd_origin_flags(
        self,
        *,
        state: str,
        completed_modules: Optional[List[str]],
        cmd_from_new_inference: Optional[bool],
        cmd_from_cached_output: Optional[bool],
    ) -> Tuple[bool, bool]:
        if cmd_from_new_inference is not None and cmd_from_cached_output is not None:
            return bool(cmd_from_new_inference), bool(cmd_from_cached_output)

        inferred_new = False
        inferred_cached = False
        if self._is_model_omega_state(state):
            completed = set(completed_modules or [])
            inferred_new = "straight_keep" in completed
            inferred_cached = not inferred_new

        if cmd_from_new_inference is not None:
            inferred_new = bool(cmd_from_new_inference)
        if cmd_from_cached_output is not None:
            inferred_cached = bool(cmd_from_cached_output)
        return inferred_new, inferred_cached

    def _resolve_omega_source(
        self,
        *,
        state: str,
        reason: str,
        image_age_ms: int,
        latest_inference_age_ms: int,
        cmd_from_cached_output: bool,
        source_override: Optional[str] = None,
    ) -> str:
        if source_override:
            return str(source_override)

        reason_s = str(reason or "")
        if reason_s.startswith("model_output_timeout"):
            return "timeout_zero"
        if reason_s in (
            "missing_image",
            "missing_image_or_timeout",
            "startup_warmup_waiting_first_image",
        ):
            if int(image_age_ms) >= 0 and (
                float(image_age_ms) / 1000.0
            ) > float(self.image_timeout_sec):
                return "timeout_zero"
            return "missing_image_zero"
        if reason_s in ("exception", "control_tick_exception", "too_many_consecutive_errors"):
            return "exception_zero"

        if state == "BOOT":
            return "boot_zero"
        if state == "TURN":
            return "turn_fixed"
        if state == "RECOVER":
            base_source = "recover_blend"
        elif state in ("STRAIGHTKEEP", "APPROACH", "PROVISIONAL_TURN"):
            base_source = "straight_keep"
        else:
            base_source = "boot_zero"

        if (
            bool(cmd_from_cached_output)
            and int(latest_inference_age_ms) >= 0
            and int(latest_inference_age_ms) >= self._stale_cache_age_threshold_ms()
        ):
            return "stale_cache"
        return base_source

    def _empty_straight_keep_trace(self) -> Dict[str, Any]:
        trace: Dict[str, Any] = {
            "straight_keep_raw_omega": None,
            "straight_keep_bias": float(self.straight_keep_bias),
            "straight_keep_scale": float(self.straight_keep_scale),
            "straight_keep_deadband": float(self.straight_keep_deadband),
            "straight_keep_after_bias": None,
            "straight_keep_after_scale": None,
            "straight_keep_after_deadband": None,
            "straight_keep_after_clip": None,
            "straight_keep_final_omega": None,
        }
        return trace

    def _calibrate_straight_keep_omega(
        self, raw_omega: float
    ) -> Tuple[float, Dict[str, Any]]:
        trace = self._empty_straight_keep_trace()
        raw_omega = float(raw_omega)
        after_bias = raw_omega + float(self.straight_keep_bias)
        after_scale = after_bias * float(self.straight_keep_scale)
        after_deadband = (
            0.0
            if abs(after_scale) < float(self.straight_keep_deadband)
            else float(after_scale)
        )
        after_clip = self._clip(
            after_deadband, -float(self.angular_clip), float(self.angular_clip)
        )
        previous_omega = (
            float(self._last_angular_z) if self._last_angular_cmd_valid else 0.0
        )
        final_omega = self._rate_limit_omega(after_clip, previous_omega)
        trace.update(
            {
                "straight_keep_raw_omega": float(raw_omega),
                "straight_keep_after_bias": float(after_bias),
                "straight_keep_after_scale": float(after_scale),
                "straight_keep_after_deadband": float(after_deadband),
                "straight_keep_after_clip": float(after_clip),
                "straight_keep_final_omega": float(final_omega),
            }
        )
        return float(final_omega), trace

    def _build_pulse_recenter_diag(
        self, *, error: float, reason: str
    ) -> Dict[str, Any]:
        return {
            "pulse_recenter_enable": bool(self.pulse_recenter_enable),
            "pulse_recenter_state": str(self._pulse_recenter_state),
            "pulse_recenter_error": float(error),
            "pulse_recenter_dir": int(self._pulse_recenter_dir),
            "pulse_recenter_step_count": int(self._pulse_recenter_step_count),
            "pulse_recenter_enter_count": int(self._pulse_recenter_enter_count),
            "pulse_recenter_exit_count": int(self._pulse_recenter_exit_count),
            "pulse_recenter_cooldown_count": int(
                self._pulse_recenter_cooldown_count
            ),
            "pulse_recenter_enter_abs": float(self.pulse_recenter_enter_abs),
            "pulse_recenter_exit_abs": float(self.pulse_recenter_exit_abs),
            "pulse_recenter_omega": float(self.pulse_recenter_omega),
            "pulse_recenter_reason": str(reason),
        }

    def _apply_pulse_recenter_policy(
        self,
        *,
        raw_omega: float,
        linear_x: float,
        image_age_ms: int,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        slow_straight_only 专用脉冲纠偏策略。

        straight_keep_bias 在这里表示模型零点标定值：
        error = raw_omega + bias。默认保持 angular.z=0，仅在误差连续
        超过进入阈值时输出短脉冲，随后进入冷却并回到零角速度直行。
        """
        del image_age_ms  # 预留给后续图像新鲜度门控；当前策略只记录控制误差。

        raw_omega = float(raw_omega)
        linear_x = float(linear_x)
        error = raw_omega + float(self.straight_keep_bias)
        angular_z = 0.0
        reason = "disabled"

        if not self.pulse_recenter_enable:
            self._pulse_recenter_state = "CENTER_HOLD"
            self._pulse_recenter_dir = 0
            self._pulse_recenter_step_count = 0
            self._pulse_recenter_enter_count = 0
            self._pulse_recenter_exit_count = 0
            self._pulse_recenter_cooldown_count = 0
            return (
                linear_x,
                0.0,
                self._build_pulse_recenter_diag(error=error, reason=reason),
            )

        if self._pulse_recenter_state not in (
            "CENTER_HOLD",
            "RECENTERING",
            "COOLDOWN",
        ):
            self._pulse_recenter_state = "CENTER_HOLD"
            self._pulse_recenter_dir = 0
            self._pulse_recenter_step_count = 0
            self._pulse_recenter_enter_count = 0
            self._pulse_recenter_exit_count = 0
            self._pulse_recenter_cooldown_count = 0

        state = self._pulse_recenter_state

        if state == "CENTER_HOLD":
            # 默认保持零角速度；只有连续达到进入阈值才触发一次纠偏脉冲。
            angular_z = 0.0
            self._pulse_recenter_dir = 0
            self._pulse_recenter_step_count = 0
            self._pulse_recenter_exit_count = 0

            if abs(error) >= float(self.pulse_recenter_enter_abs):
                self._pulse_recenter_enter_count += 1
                reason = "enter_vote"
            else:
                self._pulse_recenter_enter_count = 0
                reason = "center_hold"

            if self._pulse_recenter_enter_count >= int(
                self.pulse_recenter_enter_votes
            ):
                self._pulse_recenter_state = "RECENTERING"
                self._pulse_recenter_dir = 1 if error > 0.0 else -1
                self._pulse_recenter_step_count = 0
                self._pulse_recenter_enter_count = 0
                self._pulse_recenter_exit_count = 0
                reason = "enter_recentering"

                # 进入纠偏状态的同一帧发出首个短脉冲，减少慢模型下的响应延迟。
                angular_z = (
                    float(self._pulse_recenter_dir)
                    * float(self.pulse_recenter_omega)
                )
                self._pulse_recenter_step_count = 1

        elif state == "RECENTERING":
            recenter_dir = int(self._pulse_recenter_dir)
            if recenter_dir not in (-1, 1):
                recenter_dir = 1 if error > 0.0 else -1
                self._pulse_recenter_dir = recenter_dir

            error_sign = self._sign_of(error)
            sign_flip = (
                bool(self.pulse_recenter_stop_on_sign_flip)
                and error_sign in (-1, 1)
                and error_sign == -recenter_dir
            )

            if abs(error) <= float(self.pulse_recenter_exit_abs):
                self._pulse_recenter_exit_count += 1
            else:
                self._pulse_recenter_exit_count = 0

            should_stop = False
            if sign_flip:
                should_stop = True
                reason = "sign_flip_stop"
            elif self._pulse_recenter_exit_count >= int(
                self.pulse_recenter_exit_votes
            ):
                should_stop = True
                reason = "exit_abs_hold"
            elif self._pulse_recenter_step_count >= int(
                self.pulse_recenter_max_steps
            ):
                should_stop = True
                reason = "max_steps"
            else:
                reason = "recentering"

            if should_stop:
                angular_z = 0.0
                self._pulse_recenter_state = "COOLDOWN"
                self._pulse_recenter_cooldown_count = int(
                    self.pulse_recenter_cooldown_steps
                )
                self._pulse_recenter_dir = 0
                self._pulse_recenter_step_count = 0
                self._pulse_recenter_enter_count = 0
                self._pulse_recenter_exit_count = 0
            else:
                angular_z = recenter_dir * float(self.pulse_recenter_omega)
                self._pulse_recenter_step_count += 1

        else:  # COOLDOWN
            # 冷却期间强制零角速度，避免连续脉冲变成持续转向。
            angular_z = 0.0
            self._pulse_recenter_enter_count = 0
            self._pulse_recenter_exit_count = 0
            self._pulse_recenter_dir = 0
            self._pulse_recenter_step_count = 0
            if self._pulse_recenter_cooldown_count > 0:
                self._pulse_recenter_cooldown_count -= 1
                reason = "cooldown"
            else:
                reason = "cooldown_done"
            if self._pulse_recenter_cooldown_count <= 0:
                self._pulse_recenter_state = "CENTER_HOLD"
                self._pulse_recenter_cooldown_count = 0

        if self._pulse_recenter_state != "RECENTERING":
            angular_z = 0.0

        return (
            linear_x,
            float(angular_z),
            self._build_pulse_recenter_diag(error=error, reason=reason),
        )

    def _recover_same_direction_suppressed_omega(
        self, *, omega: float, locked_turn_dir: Optional[str]
    ) -> float:
        omega_sign = self._sign_of(float(omega))
        if omega_sign == 0:
            return 0.0
        turn_sign = self._turn_dir_to_sign(locked_turn_dir)
        if turn_sign not in (-1, 1):
            turn_sign = (
                int(self._debug_last_turn_sign)
                if self._debug_last_turn_sign in (-1, 1)
                else None
            )
        if turn_sign in (-1, 1) and omega_sign == int(turn_sign):
            return 0.0
        return float(omega)

    def _build_cmd_diag(
        self,
        *,
        now: Time,
        state: str,
        locked_turn_dir: Optional[str],
        linear_x: float,
        angular_z: float,
        reason: str,
        image_age_ms: int,
        omega_cmd_final: Optional[float] = None,
        straight_keep_trace: Optional[Dict[str, Any]] = None,
        completed_modules: Optional[List[str]] = None,
        source_override: Optional[str] = None,
        cmd_from_new_inference: Optional[bool] = None,
        cmd_from_cached_output: Optional[bool] = None,
        reused_last_cmd: bool = False,
    ) -> Dict[str, Any]:
        prev_nav_state = self._debug_last_nav_state
        if prev_nav_state == state:
            self._debug_state_step_count += 1
        else:
            self._debug_state_step_count = 1

        if state == "TURN":
            if prev_nav_state == "TURN":
                self._debug_turn_step_count += 1
            else:
                self._debug_turn_step_count = 1
        else:
            self._debug_turn_step_count = 0

        if state == "RECOVER":
            if prev_nav_state == "RECOVER":
                self._debug_recover_step_count += 1
            else:
                self._debug_recover_step_count = 1
        else:
            self._debug_recover_step_count = 0

        state_step_count = int(self._debug_state_step_count)
        turn_step_count = int(self._debug_turn_step_count) if state == "TURN" else -1
        recover_step_count = int(self._debug_recover_step_count) if state == "RECOVER" else -1
        state_step_sm = getattr(self.state_machine, "state_step", None)
        if isinstance(state_step_sm, int) and state_step_sm >= 0:
            state_step_count = int(state_step_sm)
            if state == "TURN":
                turn_step_count = int(state_step_sm)
            if state == "RECOVER":
                recover_step_count = int(state_step_sm)
        self._debug_last_nav_state = state

        last_turn_sign_before = self._debug_last_turn_sign
        omega_sign = self._sign_of(float(angular_z))
        omega_same_sign_as_last_turn: Optional[bool] = None
        if last_turn_sign_before in (-1, 1) and omega_sign != 0:
            omega_same_sign_as_last_turn = bool(omega_sign == int(last_turn_sign_before))

        locked_sign = self._turn_dir_to_sign(locked_turn_dir)
        if locked_sign in (-1, 1):
            self._debug_last_turn_dir = str(locked_turn_dir)
            self._debug_last_turn_sign = int(locked_sign)
        elif state == "TURN" and omega_sign != 0:
            self._debug_last_turn_sign = int(omega_sign)
            self._debug_last_turn_dir = self._sign_to_turn_dir(omega_sign)

        latest_inference_age_ms = self._get_cmd_inference_age_ms(state, now)
        cmd_new, cmd_cached = self._compute_cmd_origin_flags(
            state=state,
            completed_modules=completed_modules,
            cmd_from_new_inference=cmd_from_new_inference,
            cmd_from_cached_output=cmd_from_cached_output,
        )
        omega_source = self._resolve_omega_source(
            state=state,
            reason=reason,
            image_age_ms=image_age_ms,
            latest_inference_age_ms=latest_inference_age_ms,
            cmd_from_cached_output=cmd_cached,
            source_override=source_override,
        )

        diag = {
            "nav_state": str(state),
            "prev_nav_state": prev_nav_state,
            "state_step_count": int(state_step_count),
            "turn_step_count": int(turn_step_count),
            "recover_step_count": int(recover_step_count),
            "locked_turn_dir": locked_turn_dir if locked_turn_dir in ("Left", "Right") else None,
            "last_turn_dir": self._debug_last_turn_dir,
            "last_turn_sign": self._debug_last_turn_sign
            if self._debug_last_turn_sign in (-1, 1)
            else None,
            "cmd_linear_x": float(linear_x),
            "cmd_angular_z": float(angular_z),
            "omega_source": str(omega_source),
            "cmd_from_new_inference": bool(cmd_new),
            "cmd_from_cached_output": bool(cmd_cached),
            "reused_last_cmd": bool(reused_last_cmd),
            "omega_same_sign_as_last_turn": omega_same_sign_as_last_turn,
            "latest_inference_age_ms": int(latest_inference_age_ms),
            "model_output_timeout": float(self.model_output_timeout_sec),
            "image_timeout": float(self.image_timeout_sec),
            "stale_omega_suppressed": False,
            "stale_omega_before": None,
            "stale_omega_after": None,
            "max_omega_hold_sec": float(self.max_omega_hold_sec),
            "omega_stale_decay": float(self.omega_stale_decay),
            "stale_linear_hold_active": False,
            "stale_linear_hold_allowed": False,
            "stale_linear_hold_reason": "",
            "linear_hold_speed_on_stale": float(self.linear_hold_speed_on_stale),
            "max_linear_hold_sec": float(self.max_linear_hold_sec),
            "linear_hold_max_image_age_sec": float(self.linear_hold_max_image_age_sec),
        }
        if isinstance(straight_keep_trace, dict):
            diag.update(straight_keep_trace)
        else:
            diag.update(self._empty_straight_keep_trace())
        return diag

    # ---------------------------
    # 配置与导入
    # ---------------------------
    def _cfg_dict(self, key: str) -> Dict[str, Any]:
        value = self.config.get(key, {})
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _resolve_topic_value(
        *, launch_value: str, yaml_value: str, default_value: str
    ) -> Tuple[str, str]:
        launch_v = str(launch_value or "").strip()
        if launch_v:
            return launch_v, "launch"

        yaml_v = str(yaml_value or "").strip()
        if yaml_v:
            return yaml_v, "yaml"

        return default_value, "default"

    @staticmethod
    def _dedupe_paths(paths: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for p in paths:
            key = os.path.normcase(os.path.normpath(p))
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    @staticmethod
    def _format_missing_path_message(raw_path: str, tried_candidates: List[str]) -> str:
        lines = [
            f"raw_path={raw_path}",
            "tried_candidates:",
        ]
        if tried_candidates:
            lines.extend(f"  - {p}" for p in tried_candidates)
        else:
            lines.append("  - <none>")
        return "\n".join(lines)

    def _detect_repo_root(self, config_path: Optional[str]) -> Optional[str]:
        candidates: List[Path] = []

        env_repo_root = os.path.expandvars(os.path.expanduser(os.environ.get("SNN_ROOT", "")))
        if env_repo_root:
            candidates.append(Path(env_repo_root))

        if config_path:
            cfg = Path(config_path).resolve()
            if cfg.parent.name == "configs":
                candidates.append(cfg.parent.parent)
            candidates.append(cfg.parent)

        this_file = Path(__file__).resolve()
        candidates.extend(list(this_file.parents))

        cwd = Path.cwd().resolve()
        candidates.append(cwd)
        candidates.extend(list(cwd.parents))

        seen = set()
        for p in candidates:
            p_resolved = p.resolve()
            key = str(p_resolved).lower()
            if key in seen:
                continue
            seen.add(key)
            if self._looks_like_repo_root(p_resolved):
                return str(p_resolved)
        return None

    def _resolve_path_with_candidates(
        self, path_raw: str, base_dir: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        raw = str(path_raw or "").strip()
        if not raw:
            return "", []

        expanded = os.path.expandvars(os.path.expanduser(raw))
        if os.path.isabs(expanded):
            candidates = [os.path.abspath(expanded)]
        else:
            candidates = []
            if self.repo_root:
                candidates.append(os.path.abspath(os.path.join(self.repo_root, expanded)))
            if base_dir:
                candidates.append(os.path.abspath(os.path.join(base_dir, expanded)))
            candidates.append(os.path.abspath(os.path.join(os.getcwd(), expanded)))
            candidates = self._dedupe_paths(candidates)

        for c in candidates:
            if os.path.exists(c):
                return c, candidates
        return candidates[0], candidates

    def _resolve_config_path(self, cfg_path_raw: str) -> str:
        if not cfg_path_raw:
            cfg_path_raw = "configs/hierarchical_nav_robot_v1.yaml"

        resolved, candidates = self._resolve_path_with_candidates(cfg_path_raw)
        if os.path.isfile(resolved):
            return str(Path(resolved).resolve())

        raise FileNotFoundError(
            "config_path not found.\n"
            + self._format_missing_path_message(cfg_path_raw, candidates)
        )

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("配置文件根节点必须为 dict: %s" % path)
        return cfg

    @staticmethod
    def _looks_like_repo_root(path_obj: Path) -> bool:
        return (
            (path_obj / "inference" / "corridor_module_infer.py").is_file()
            and (path_obj / "controllers" / "hierarchical_state_machine.py").is_file()
        )

    def _prepare_repo_import_path(self, config_path: str) -> None:
        """
        尝试将仓库根目录加入 sys.path，同时刷新路径解析时使用的 repo_root。
        """
        detected = self._detect_repo_root(config_path=config_path)
        if detected:
            self.repo_root = detected
            if detected not in sys.path:
                sys.path.insert(0, detected)
            self.get_logger().info("Repository root resolved for imports: %s" % detected)
            return

        self.get_logger().warn("未自动定位到仓库根目录，后续将尝试直接导入 inference/controllers。")

    def _log_import_failure(self, import_target: str, exc: Exception) -> None:
        self.get_logger().error("Python executable: %s" % sys.executable)
        self.get_logger().error("sys.path[0:5]: %s" % repr(sys.path[:5]))
        self.get_logger().error("Import %s failed: %r" % (import_target, exc))
        self.get_logger().error(
            "Traceback for failed import %s:\\n%s"
            % (import_target, traceback.format_exc())
        )

    def _import_infer_classes(self):
        try:
            from inference.corridor_module_infer import (  # type: ignore
                ApproachTriggerInfer,
                JunctionLRInfer,
                Stage3Infer,
                StraightKeepInfer,
            )
        except Exception as exc:  # pragma: no cover
            self._log_import_failure("inference.corridor_module_infer", exc)
            raise
        return Stage3Infer, JunctionLRInfer, StraightKeepInfer, ApproachTriggerInfer

    def _import_state_machine_class(self):
        try:
            from controllers.hierarchical_state_machine import (  # type: ignore
                HierarchicalNavigatorStateMachine,
            )
        except Exception as exc:  # pragma: no cover
            self._log_import_failure("controllers.hierarchical_state_machine", exc)
            raise
        return HierarchicalNavigatorStateMachine

    def _build_state_machine(self):
        # 组装 kwargs：state_machine 为主，turn_control/straight_keep 兜底补充
        sm_kwargs: Dict[str, Any] = dict(self.state_machine_cfg)
        sm_kwargs.setdefault(
            "left_turn_omega", float(self.turn_control_cfg.get("left_omega", 0.5))
        )
        sm_kwargs.setdefault(
            "right_turn_omega", float(self.turn_control_cfg.get("right_omega", -0.5))
        )
        sm_kwargs.setdefault(
            "use_fixed_turn_rate",
            bool(self.turn_control_cfg.get("use_fixed_turn_rate", True)),
        )
        sm_kwargs.setdefault(
            "max_turn_steps", int(self.turn_control_cfg.get("max_turn_steps", 20))
        )
        sm_kwargs.setdefault(
            "omega_clip", float(self.straight_keep_cfg.get("omega_clip", 1.2))
        )
        sm_kwargs.setdefault("use_clip", bool(self.straight_keep_cfg.get("use_clip", True)))

        # 过滤未知参数，兼容后续配置字段扩展
        init_sig = inspect.signature(self.HierarchicalNavigatorStateMachine.__init__)
        allowed = {k for k in init_sig.parameters.keys() if k != "self"}
        filtered = {k: v for k, v in sm_kwargs.items() if k in allowed}

        sm = self.HierarchicalNavigatorStateMachine(**filtered)
        return sm

    def _load_models(self) -> None:
        """加载四个推理模块。失败时保留 None，运行期进入安全兜底。"""
        model_specs = {
            "stage3": ("stage3_ckpt", self.Stage3Infer),
            "junction_lr": ("junction_lr_ckpt", self.JunctionLRInfer),
            "straight_keep": ("straight_keep_ckpt", self.StraightKeepInfer),
            "approach_trigger": ("approach_trigger_ckpt", self.ApproachTriggerInfer),
        }
        for name, (ckpt_key, cls_obj) in model_specs.items():
            ckpt_raw = str(self.models_cfg.get(ckpt_key, "")).strip()
            if not ckpt_raw:
                self.models[name] = None
                self.get_logger().error("模型配置缺失: models.%s" % ckpt_key)
                continue

            ckpt_path, tried_candidates = self._resolve_path_with_candidates(
                ckpt_raw,
                base_dir=self.config_dir,
            )
            if not os.path.isfile(ckpt_path):
                self.models[name] = None
                self.get_logger().error(
                    "模型 checkpoint 不存在: models.%s\n%s"
                    % (ckpt_key, self._format_missing_path_message(ckpt_raw, tried_candidates))
                )
                continue

            try:
                self.models[name] = cls_obj(ckpt_path=ckpt_path, device=None)
                # 在线串流推理前，显式重置一次内部状态
                if hasattr(self.models[name], "reset_state"):
                    self.models[name].reset_state()
                self.get_logger().info("Loaded model[%s]: %s" % (name, ckpt_path))
            except Exception as exc:
                self.models[name] = None
                self.get_logger().error(
                    "模型加载失败[%s]: %s (%s)" % (name, ckpt_path, str(exc))
                )

    # ---------------------------
    # 图像处理
    # ---------------------------
    def _image_callback(self, msg: Image) -> None:
        """
        latest-only 图像缓存：
        1) 仅保存最新帧 + 时间戳；
        2) 不做推理；
        3) 旧帧直接覆盖。
        """
        recv_time = self.get_clock().now()
        try:
            image_rgb = self._to_rgb_image(msg)
            header_time = Time.from_msg(msg.header.stamp)
            if header_time.nanoseconds <= 0:
                header_time = recv_time
            should_log_first_image = False
            with self._img_lock:
                self._latest_image = image_rgb
                self._latest_image_time = recv_time
                self._latest_image_header_time = header_time
                self._latest_image_receive_time = recv_time
                self._latest_image_stamp = header_time
                self._image_rx_count += 1
                self._has_received_first_image = True
                if not self._first_image_logged:
                    self._first_image_logged = True
                    should_log_first_image = True
            if should_log_first_image:
                self.get_logger().info(
                    "First image received. topic=%s, header_stamp=%s, receive_time=%s"
                    % (
                        self.image_topic,
                        self._format_time_stamp(header_time),
                        self._format_time_stamp(recv_time),
                    )
                )
        except Exception as exc:
            self.get_logger().error("图像转换失败: %s" % str(exc))

    def _to_rgb_image(self, msg: Image) -> np.ndarray:
        """
        图像转换优先使用 cv_bridge；
        若 cv_bridge 不可用或转换失败，则走手工解析 fallback。
        """
        if self._bridge is not None:
            try:
                img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                return np.ascontiguousarray(img)
            except Exception as exc:
                if not self._bridge_warned:
                    self.get_logger().warn(
                        "cv_bridge 转换失败，切换到手工解析路径: %s" % str(exc)
                    )
                    self._bridge_warned = True

        # fallback：支持常见编码
        enc = str(msg.encoding).lower()
        h, w = int(msg.height), int(msg.width)
        if h <= 0 or w <= 0:
            raise ValueError("非法图像尺寸: h=%d, w=%d" % (h, w))
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if enc in ("rgb8", "bgr8"):
            c = 3
        elif enc in ("mono8", "8uc1"):
            c = 1
        elif enc in ("rgba8", "bgra8"):
            c = 4
        else:
            raise ValueError("不支持的图像编码: %s" % msg.encoding)

        if msg.step <= 0:
            raise ValueError("图像 step 非法: %d" % int(msg.step))
        min_bytes = h * int(msg.step)
        if data.size < min_bytes:
            raise ValueError("图像数据长度不足: got=%d, need=%d" % (data.size, min_bytes))

        rows = data[:min_bytes].reshape((h, int(msg.step)))
        need = w * c
        rows = rows[:, :need]
        img = rows.reshape((h, w, c))

        if c == 1:
            img = np.repeat(img, 3, axis=2)
        elif enc == "bgr8":
            img = img[:, :, ::-1]
        elif enc == "rgba8":
            img = img[:, :, :3]
        elif enc == "bgra8":
            img = img[:, :, [2, 1, 0]]

        return np.ascontiguousarray(img)

    # ---------------------------
    # 主控制循环
    # ---------------------------
    def _control_tick(self) -> None:
        self._tick_count += 1
        self._active_scheduled_module_this_tick = None
        now = self.get_clock().now()

        state_now = self._get_state_name()
        locked_now = self._get_locked_turn_dir()

        run_flags = {
            "stage3": False,
            "junction_lr": False,
            "straight_keep": False,
            "approach_trigger": False,
        }
        outputs = self._get_cached_outputs()
        completed_modules: List[str] = []
        startup_elapsed_sec = self._age_sec(self._node_start_time, now)
        startup_warmup_active = (
            startup_elapsed_sec is not None
            and startup_elapsed_sec <= self.startup_warmup_sec
        )

        try:
            tick_had_exception, completed_modules = self._collect_finished_inference_results(
                now=now
            )
            outputs = self._get_cached_outputs()
            if tick_had_exception:
                self._consecutive_errors += 1
            else:
                self._consecutive_errors = 0
            if self.runtime_mode == "slow_straight_only":
                state_now = self._slow_straight_state_name(startup_warmup_active)
                locked_now = None

            image_ok, image_age_ms = self._get_latest_image_status(now)
            if tick_had_exception:
                reason = "model_inference_exception"
                cmd_diag = self._build_cmd_diag(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    source_override="exception_zero",
                    cmd_from_new_inference=False,
                    cmd_from_cached_output=False,
                )
                self._publish_state(state_now)
                self._publish_debug(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    outputs=outputs,
                    run_flags=run_flags,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    image_received_ok=image_ok,
                    cmd_diag=cmd_diag,
                )
                self.publish_zero_twist(reason)
                return

            if not image_ok:
                with self._img_lock:
                    has_received_first_image = bool(self._has_received_first_image)

                if startup_warmup_active and (not has_received_first_image):
                    reason = "startup_warmup_waiting_first_image"
                    cmd_diag = self._build_cmd_diag(
                        now=now,
                        state=state_now,
                        locked_turn_dir=locked_now,
                        linear_x=0.0,
                        angular_z=0.0,
                        reason=reason,
                        image_age_ms=image_age_ms,
                        source_override="missing_image_zero",
                        cmd_from_new_inference=False,
                        cmd_from_cached_output=False,
                    )
                    self._publish_state(state_now)
                    self._publish_debug(
                        now=now,
                        state=state_now,
                        locked_turn_dir=locked_now,
                        linear_x=0.0,
                        angular_z=0.0,
                        outputs=outputs,
                        run_flags=run_flags,
                        reason=reason,
                        image_age_ms=image_age_ms,
                        image_received_ok=False,
                        cmd_diag=cmd_diag,
                    )
                    self.publish_zero_twist(reason)
                    return

                reason = "missing_image_or_timeout"
                self._maybe_log_missing_image(now=now, image_age_ms=image_age_ms)
                cmd_diag = self._build_cmd_diag(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    cmd_from_new_inference=False,
                    cmd_from_cached_output=False,
                )
                self._publish_state(state_now)
                self._publish_debug(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    outputs=outputs,
                    run_flags=run_flags,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    image_received_ok=False,
                    cmd_diag=cmd_diag,
                )
                self.publish_zero_twist(reason)
                return

            image_np = self._get_latest_image_copy()
            if image_np is None:
                reason = "missing_image"
                cmd_diag = self._build_cmd_diag(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    source_override="missing_image_zero",
                    cmd_from_new_inference=False,
                    cmd_from_cached_output=False,
                )
                self._publish_state(state_now)
                self._publish_debug(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    outputs=outputs,
                    run_flags=run_flags,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    image_received_ok=False,
                    cmd_diag=cmd_diag,
                )
                self.publish_zero_twist(reason)
                return

            schedule_flags = self._decide_schedule(state_now, locked_now)
            run_flags, submit_had_exception = self._submit_inference_jobs(
                image_np=image_np,
                schedule_flags=schedule_flags,
            )
            outputs = self._get_cached_outputs()
            if submit_had_exception:
                self._consecutive_errors += 1
                reason = "model_submit_exception"
                cmd_diag = self._build_cmd_diag(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    source_override="exception_zero",
                    cmd_from_new_inference=False,
                    cmd_from_cached_output=False,
                )
                self._publish_state(state_now)
                self._publish_debug(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    outputs=outputs,
                    run_flags=run_flags,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    image_received_ok=True,
                    cmd_diag=cmd_diag,
                )
                self.publish_zero_twist(reason)
                return

            if self._consecutive_errors >= self.max_consecutive_errors:
                reason = "too_many_consecutive_errors"
                self.get_logger().error(
                    "Too many consecutive errors: %d >= %d, publishing zero command."
                    % (self._consecutive_errors, self.max_consecutive_errors)
                )
                cmd_diag = self._build_cmd_diag(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    source_override="exception_zero",
                    cmd_from_new_inference=False,
                    cmd_from_cached_output=False,
                )
                self._publish_state(state_now)
                self._publish_debug(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    outputs=outputs,
                    run_flags=run_flags,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    image_received_ok=True,
                    cmd_diag=cmd_diag,
                )
                self.publish_zero_twist(reason)
                return

            if self.runtime_mode == "slow_straight_only":
                self._control_tick_slow_straight_only(
                    now=now,
                    outputs=outputs,
                    run_flags=run_flags,
                    completed_modules=completed_modules,
                    startup_warmup_active=startup_warmup_active,
                    image_age_ms=image_age_ms,
                )
                return

            debug_reason = "startup_warmup" if startup_warmup_active else "ok"

            timeout_modules = []
            if not startup_warmup_active:
                timeout_modules = self._check_model_timeout_modules(
                    state=state_now,
                    locked_turn_dir=locked_now,
                    now=now,
                )
            if timeout_modules:
                reason = "model_output_timeout:%s" % ",".join(timeout_modules)
                self.get_logger().error(
                    "Model output timeout: %s" % ",".join(timeout_modules)
                )
                cmd_diag = self._build_cmd_diag(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    source_override="timeout_zero",
                    cmd_from_new_inference=False,
                    cmd_from_cached_output=False,
                )
                self._publish_state(state_now)
                self._publish_debug(
                    now=now,
                    state=state_now,
                    locked_turn_dir=locked_now,
                    linear_x=0.0,
                    angular_z=0.0,
                    outputs=outputs,
                    run_flags=run_flags,
                    reason=reason,
                    image_age_ms=image_age_ms,
                    image_received_ok=True,
                    cmd_diag=cmd_diag,
                )
                self.publish_zero_twist(reason)
                return

            sm_out = self.state_machine.update(
                {
                    "stage3": outputs["stage3"],
                    "junction_lr": outputs["junction_lr"],
                    "straight_keep": outputs["straight_keep"],
                    "approach_trigger": outputs["approach_trigger"],
                }
            )
            state_new = str(sm_out.get("state", state_now))
            locked_new = sm_out.get("locked_turn_dir", locked_now)
            omega_sm = self._safe_float(sm_out.get("omega_cmd_final", 0.0), 0.0)
            straight_keep_raw_omega = self._safe_float(
                outputs.get("straight_keep", {}).get("omega_cmd_raw", omega_sm),
                omega_sm,
            )

            linear_x, angular_z, straight_keep_trace = self._compose_control_cmd(
                state=state_new,
                locked_turn_dir=locked_new,
                omega_cmd_final=omega_sm,
                straight_keep_raw_omega=straight_keep_raw_omega,
            )
            trigger_pred_for_hold = self._parse_trigger_pred(
                outputs.get("approach_trigger", {})
            )
            stage3_pred_for_hold = self._parse_stage3_pred(outputs.get("stage3", {}))
            (
                linear_x,
                angular_z,
                source_override,
                stale_omega_diag,
                cmd_from_new,
                cmd_from_cached,
            ) = self._apply_stale_omega_policy(
                now=now,
                state=state_new,
                linear_x=linear_x,
                angular_z=angular_z,
                reason=debug_reason,
                image_age_ms=image_age_ms,
                trigger_pred=trigger_pred_for_hold,
                stage3_pred=stage3_pred_for_hold,
                completed_modules=completed_modules,
            )
            cmd_diag = self._build_cmd_diag(
                now=now,
                state=state_new,
                locked_turn_dir=locked_new,
                linear_x=linear_x,
                angular_z=angular_z,
                reason=debug_reason,
                image_age_ms=image_age_ms,
                omega_cmd_final=omega_sm,
                straight_keep_trace=straight_keep_trace,
                completed_modules=completed_modules,
                source_override=source_override,
                cmd_from_new_inference=cmd_from_new,
                cmd_from_cached_output=cmd_from_cached,
                reused_last_cmd=False,
            )
            cmd_diag.update(stale_omega_diag)

            cmd = Twist()
            cmd.linear.x = float(linear_x)
            cmd.angular.z = float(angular_z)

            self._publish_state(state_new)
            self._publish_debug(
                now=now,
                state=state_new,
                locked_turn_dir=locked_new,
                linear_x=linear_x,
                angular_z=angular_z,
                outputs=outputs,
                run_flags=run_flags,
                reason=debug_reason,
                image_age_ms=image_age_ms,
                image_received_ok=True,
                sm_out=sm_out,
                cmd_diag=cmd_diag,
            )
            self.cmd_pub.publish(cmd)
            self._record_published_cmd(cmd)
        except Exception as exc:
            self._consecutive_errors += 1
            self.get_logger().error(
                "control_tick ??: %s\n%s" % (str(exc), traceback.format_exc())
            )
            exc_image_age_ms = self._age_ms(
                self._latest_image_receive_time or self._latest_image_time, now
            )
            cmd_diag = self._build_cmd_diag(
                now=now,
                state=state_now,
                locked_turn_dir=locked_now,
                linear_x=0.0,
                angular_z=0.0,
                reason="exception",
                image_age_ms=exc_image_age_ms,
                source_override="exception_zero",
                cmd_from_new_inference=False,
                cmd_from_cached_output=False,
            )
            self._publish_state(state_now)
            self._publish_debug(
                now=now,
                state=state_now,
                locked_turn_dir=locked_now,
                linear_x=0.0,
                angular_z=0.0,
                outputs=outputs,
                run_flags=run_flags,
                reason="exception",
                image_age_ms=exc_image_age_ms,
                image_received_ok=None,
                extra={"exception": str(exc)},
                cmd_diag=cmd_diag,
            )
            self.publish_zero_twist("control_tick_exception")

    def _control_tick_slow_straight_only(
        self,
        *,
        now: Time,
        outputs: Dict[str, Dict[str, Any]],
        run_flags: Dict[str, bool],
        completed_modules: List[str],
        startup_warmup_active: bool,
        image_age_ms: int,
    ) -> None:
        """慢模型安全直行模式：只依赖 straight_keep，绕开路口状态机。"""
        locked_turn_dir: Optional[str] = None

        with self._module_lock:
            straight_keep_update = self.module_cache["straight_keep"].last_update_time

        has_first_output = straight_keep_update is not None
        waiting_first_output = not has_first_output

        def publish_zero(
            *,
            zero_state: str,
            reason: str,
            safety_level: str,
            hard_stop_reason: str = "",
        ) -> None:
            cmd_diag = self._build_cmd_diag(
                now=now,
                state=zero_state,
                locked_turn_dir=locked_turn_dir,
                linear_x=0.0,
                angular_z=0.0,
                reason=reason,
                image_age_ms=image_age_ms,
                source_override="timeout_zero"
                if safety_level == "HARD_STOP"
                else "boot_zero",
                cmd_from_new_inference=False,
                cmd_from_cached_output=False,
            )
            cmd_diag.update(
                {
                    "slow_safe_waiting_first_model_output": bool(waiting_first_output),
                    "safety_level": str(safety_level),
                    "hard_stop_reason": str(hard_stop_reason),
                }
            )
            self._publish_state(zero_state)
            self._publish_debug(
                now=now,
                state=zero_state,
                locked_turn_dir=locked_turn_dir,
                linear_x=0.0,
                angular_z=0.0,
                outputs=outputs,
                run_flags=run_flags,
                reason=reason,
                image_age_ms=image_age_ms,
                image_received_ok=True,
                cmd_diag=cmd_diag,
            )
            self.publish_zero_twist(reason)

        if startup_warmup_active:
            publish_zero(
                zero_state="BOOT",
                reason="startup_warmup",
                safety_level="WAITING_FIRST_MODEL"
                if waiting_first_output
                else "NORMAL",
            )
            return

        if waiting_first_output:
            publish_zero(
                zero_state="BOOT",
                reason="waiting_first_straight_keep_output",
                safety_level="WAITING_FIRST_MODEL",
            )
            return

        straight_keep_age_ms = self._age_ms(straight_keep_update, now)
        if (
            int(straight_keep_age_ms) < 0
            or (float(straight_keep_age_ms) / 1000.0)
            > float(self.model_output_timeout_sec)
        ):
            reason = "model_output_timeout:straight_keep"
            publish_zero(
                zero_state="STRAIGHTKEEP",
                reason=reason,
                safety_level="HARD_STOP",
                hard_stop_reason=reason,
            )
            return

        omega_raw = self._safe_float(
            outputs.get("straight_keep", {}).get("omega_cmd_raw", 0.0), 0.0
        )
        pulse_diag: Dict[str, Any] = {}
        cmd_from_new, cmd_from_cached = self._compute_cmd_origin_flags(
            state="STRAIGHTKEEP",
            completed_modules=completed_modules,
            cmd_from_new_inference=None,
            cmd_from_cached_output=None,
        )

        if self.pulse_recenter_enable:
            # pulse 模式绕开连续角速度和 stale 衰减：非新推理 tick 必须零角速度。
            linear_x = float(self.linear_speed_map.get("STRAIGHTKEEP", 0.0))
            if cmd_from_new:
                linear_x, angular_z, pulse_diag = self._apply_pulse_recenter_policy(
                    raw_omega=omega_raw,
                    linear_x=linear_x,
                    image_age_ms=image_age_ms,
                )
            else:
                angular_z = 0.0
                pulse_error = float(omega_raw) + float(self.straight_keep_bias)
                pulse_reason = "cached_zero_omega"
                if self._pulse_recenter_state == "COOLDOWN":
                    if self._pulse_recenter_cooldown_count > 0:
                        self._pulse_recenter_cooldown_count -= 1
                        pulse_reason = "cached_cooldown"
                    if self._pulse_recenter_cooldown_count <= 0:
                        self._pulse_recenter_state = "CENTER_HOLD"
                        self._pulse_recenter_cooldown_count = 0
                        pulse_reason = "cached_cooldown_done"
                pulse_diag = self._build_pulse_recenter_diag(
                    error=pulse_error,
                    reason=pulse_reason,
                )

            straight_keep_trace = self._empty_straight_keep_trace()
            pulse_error = float(omega_raw) + float(self.straight_keep_bias)
            straight_keep_trace.update(
                {
                    "straight_keep_raw_omega": float(omega_raw),
                    "straight_keep_after_bias": float(pulse_error),
                    "straight_keep_after_scale": float(pulse_error),
                    "straight_keep_after_deadband": float(angular_z),
                    "straight_keep_after_clip": float(angular_z),
                    "straight_keep_final_omega": float(angular_z),
                }
            )

            pulse_state = str(
                pulse_diag.get("pulse_recenter_state", self._pulse_recenter_state)
            )
            if abs(float(angular_z)) > 1e-9:
                source_override = "pulse_recenter"
            elif pulse_state == "COOLDOWN":
                source_override = "pulse_recenter_cooldown"
            elif pulse_state == "CENTER_HOLD":
                source_override = "pulse_center_hold"
            else:
                source_override = "pulse_cached_zero"
            stale_omega_diag = self._empty_stale_omega_diag()
        else:
            linear_x, angular_z, straight_keep_trace = self._compose_control_cmd(
                state="STRAIGHTKEEP",
                locked_turn_dir=locked_turn_dir,
                omega_cmd_final=omega_raw,
                straight_keep_raw_omega=omega_raw,
            )
            (
                linear_x,
                angular_z,
                source_override,
                stale_omega_diag,
                cmd_from_new,
                cmd_from_cached,
            ) = self._apply_stale_omega_policy(
                now=now,
                state="STRAIGHTKEEP",
                linear_x=linear_x,
                angular_z=angular_z,
                reason="ok",
                image_age_ms=image_age_ms,
                # slow_straight_only 下不让 trigger/stage3 阻塞低速保持。
                trigger_pred="Straight",
                stage3_pred="Approach",
                completed_modules=completed_modules,
            )
        hard_stop_reason = ""
        if bool(stale_omega_diag.get("stale_linear_hold_active", False)):
            safety_level = "STALE_MODEL_CREEP"
        elif (
            bool(stale_omega_diag.get("stale_omega_suppressed", False))
            and bool(cmd_from_cached)
            and abs(float(linear_x)) < 1e-9
        ):
            safety_level = "HARD_STOP"
            hard_stop_reason = "stale_straight_keep_hold_blocked:%s" % str(
                stale_omega_diag.get("stale_linear_hold_reason", "")
            )
        else:
            safety_level = "NORMAL"
        cmd_diag = self._build_cmd_diag(
            now=now,
            state="STRAIGHTKEEP",
            locked_turn_dir=locked_turn_dir,
            linear_x=linear_x,
            angular_z=angular_z,
            reason="ok",
            image_age_ms=image_age_ms,
            omega_cmd_final=omega_raw,
            straight_keep_trace=straight_keep_trace,
            completed_modules=completed_modules,
            source_override=source_override,
            cmd_from_new_inference=cmd_from_new,
            cmd_from_cached_output=cmd_from_cached,
            reused_last_cmd=False,
        )
        cmd_diag.update(stale_omega_diag)
        if pulse_diag:
            cmd_diag.update(pulse_diag)
        cmd_diag.update(
            {
                "slow_safe_waiting_first_model_output": False,
                "safety_level": safety_level,
                "hard_stop_reason": hard_stop_reason,
            }
        )

        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)

        self._publish_state("STRAIGHTKEEP")
        self._publish_debug(
            now=now,
            state="STRAIGHTKEEP",
            locked_turn_dir=locked_turn_dir,
            linear_x=linear_x,
            angular_z=angular_z,
            outputs=outputs,
            run_flags=run_flags,
            reason="ok",
            image_age_ms=image_age_ms,
            image_received_ok=True,
            cmd_diag=cmd_diag,
        )
        self.cmd_pub.publish(cmd)
        self._record_published_cmd(cmd)

    def _is_stride_tick(self, stride: int) -> bool:
        s = max(1, int(stride))
        # 第一个 tick 即命中（tick=1 -> 0 % s == 0）
        return ((self._tick_count - 1) % s) == 0

    def _count_busy_modules(self) -> int:
        with self._module_lock:
            return sum(1 for m in self._module_names if self.module_cache[m].busy)

    def _any_model_busy(self) -> bool:
        return self._count_busy_modules() > 0

    def _slow_straight_state_name(self, startup_warmup_active: bool) -> str:
        """slow_straight_only 的对外状态只在 BOOT 和 STRAIGHTKEEP 间切换。"""
        if startup_warmup_active:
            return "BOOT"
        with self._module_lock:
            has_first_output = (
                self.module_cache["straight_keep"].last_update_time is not None
            )
        return "STRAIGHTKEEP" if has_first_output else "BOOT"

    def _decide_schedule(
        self, state: str, locked_turn_dir: Optional[str]
    ) -> Dict[str, bool]:
        """
        state_conditioned_v2（必须实现）：
        - STRAIGHTKEEP: straight_keep/trigger 每步; stage3 按 stride; junction 不跑
        - APPROACH: stage3/junction/straight_keep 每步
        - PROVISIONAL_TURN: stage3/junction 每步; straight_keep 默认复用
        - TURN: junction 在锁定后停止; stage3 按 stride; straight_keep/trigger 不跑
        - RECOVER: straight_keep 每步; stage3 按 stride; junction/trigger 不跑
        """
        run = {
            "stage3": False,
            "junction_lr": False,
            "straight_keep": False,
            "approach_trigger": False,
        }

        if self.runtime_mode == "slow_straight_only":
            run["straight_keep"] = self._is_stride_tick(self.straight_keep_stride)
            return run

        if state == "STRAIGHTKEEP":
            run["straight_keep"] = self._is_stride_tick(self.straight_keep_stride)
            run["approach_trigger"] = self._is_stride_tick(self.trigger_stride)
            run["stage3"] = self._is_stride_tick(self.stage3_probe_stride)
        elif state == "APPROACH":
            run["stage3"] = True
            run["junction_lr"] = True
            run["straight_keep"] = True
        elif state == "PROVISIONAL_TURN":
            run["stage3"] = True
            run["junction_lr"] = True
            # straight_keep 默认复用缓存（不强制运行）
        elif state == "TURN":
            run["stage3"] = self._is_stride_tick(self.turn_stage3_stride)
            if self.disable_junction_after_lock and locked_turn_dir in ("Left", "Right"):
                run["junction_lr"] = False
            else:
                run["junction_lr"] = self._is_stride_tick(self.junction_probe_stride)
        elif state == "RECOVER":
            run["straight_keep"] = True
            run["stage3"] = self._is_stride_tick(self.recover_stage3_stride)
        else:  # BOOT/未知状态：保守预热
            run["straight_keep"] = self._is_stride_tick(self.straight_keep_stride)
            run["approach_trigger"] = self._is_stride_tick(self.trigger_stride)
            run["stage3"] = self._is_stride_tick(self.stage3_probe_stride)

        return run

    def _collect_finished_inference_results(self, now: Time) -> Tuple[bool, List[str]]:
        """
        仅收集已经完成的 future，不阻塞控制线程。
        """
        completed_jobs: List[Tuple[str, Future]] = []
        with self._module_lock:
            for module_name in self._module_names:
                cache = self.module_cache[module_name]
                if cache.busy and cache.future is not None and cache.future.done():
                    completed_jobs.append((module_name, cache.future))

        had_exception = False
        completed_modules: List[str] = []
        for module_name, future in completed_jobs:
            try:
                out, latency_ms, finish_wall_time = future.result()
                if not isinstance(out, dict):
                    raise TypeError("%s 输出不是 dict" % module_name)
                with self._module_lock:
                    cache = self.module_cache[module_name]
                    cache.last_output = out
                    cache.last_latency_ms = float(latency_ms)
                    cache.last_finish_wall_time = float(finish_wall_time)
                    cache.last_update_time = now
                    cache.busy = False
                    cache.future = None
                    cache.success_count += 1
                completed_modules.append(module_name)
            except Exception as exc:
                had_exception = True
                with self._module_lock:
                    cache = self.module_cache[module_name]
                    cache.busy = False
                    cache.future = None
                    cache.exception_count += 1
                self.get_logger().error(
                    "异步模型运行异常[%s]: %s\n%s"
                    % (module_name, str(exc), traceback.format_exc())
                )

        return had_exception, completed_modules

    def _submit_inference_jobs(
        self, image_np: np.ndarray, schedule_flags: Dict[str, bool]
    ) -> Tuple[Dict[str, bool], bool]:
        """
        按调度策略提交后台推理任务：
        - busy 的模块不重复提交
        - 仅提交，不等待完成
        """
        ran_flags: Dict[str, bool] = {
            "stage3": False,
            "junction_lr": False,
            "straight_keep": False,
            "approach_trigger": False,
        }
        had_exception = False
        submitted_this_tick = 0
        priority = (
            ["straight_keep"]
            if self.runtime_mode == "slow_straight_only"
            else ["straight_keep", "approach_trigger", "stage3", "junction_lr"]
        )
        max_submits_this_tick = min(
            int(self.max_submits_per_tick), 1 if self.single_active_model else 99
        )

        if self.single_active_model and self._any_model_busy():
            return ran_flags, False
        if self._count_busy_modules() >= int(self.global_max_inflight_models):
            return ran_flags, False

        for module_name in priority:
            if not schedule_flags.get(module_name, False):
                continue
            if submitted_this_tick >= max_submits_this_tick:
                break
            model = self.models.get(module_name)
            if model is None:
                continue

            should_submit = False
            with self._module_lock:
                busy_count = sum(
                    1 for m in self._module_names if self.module_cache[m].busy
                )
                if self.single_active_model and busy_count > 0:
                    break
                if busy_count >= int(self.global_max_inflight_models):
                    break
                cache = self.module_cache[module_name]
                if not cache.busy:
                    cache.busy = True
                    cache.last_run_step = self._tick_count
                    cache.last_start_wall_time = time.perf_counter()
                    should_submit = True

            if not should_submit:
                continue

            # 线程池任务使用同一时刻快照，避免读取正在变化的 latest image。
            image_snapshot = np.ascontiguousarray(image_np.copy())
            try:
                future = self._infer_executor.submit(
                    self._run_model,
                    module_name,
                    image_snapshot,
                )
            except Exception as exc:
                had_exception = True
                with self._module_lock:
                    cache = self.module_cache[module_name]
                    cache.busy = False
                    cache.future = None
                    cache.exception_count += 1
                self.get_logger().error(
                    "提交异步推理失败[%s]: %s" % (module_name, str(exc))
                )
                continue

            with self._module_lock:
                cache = self.module_cache[module_name]
                cache.future = future
            ran_flags[module_name] = True
            submitted_this_tick += 1
            if self._active_scheduled_module_this_tick is None:
                self._active_scheduled_module_this_tick = module_name
            if self.single_active_model:
                break

        return ran_flags, had_exception

    def _get_cached_outputs(self) -> Dict[str, Dict[str, Any]]:
        outputs: Dict[str, Dict[str, Any]] = {}
        with self._module_lock:
            for module_name in self._module_names:
                out = self.module_cache[module_name].last_output
                if not isinstance(out, dict):
                    out = self._default_output(module_name)
                outputs[module_name] = out
        return outputs

    def _run_model(
        self, module_name: str, image_np: np.ndarray
    ) -> Tuple[Dict[str, Any], float, float]:
        model = self.models.get(module_name)
        if model is None:
            raise RuntimeError("模型未加载: %s" % module_name)

        # 优先走 predict()
        start_wall_time = time.perf_counter()
        if hasattr(model, "predict"):
            out = model.predict(image_np)
        elif callable(model):
            out = model(image_np)
        else:
            raise RuntimeError("model object is not callable: %s" % module_name)
        finish_wall_time = time.perf_counter()
        latency_ms = (finish_wall_time - start_wall_time) * 1000.0
        return out, float(latency_ms), float(finish_wall_time)

    def _default_output(self, module_name: str) -> Dict[str, Any]:
        if module_name == "stage3":
            return {
                "pred_stage": "Approach",
                "pred_id": 0,
                "probs": {"Approach": 1.0, "Turn": 0.0, "Recover": 0.0},
                "confidence": 0.0,
            }
        if module_name == "junction_lr":
            return {
                "pred_label": "",
                "pred_id": -1,
                "probs": {},
                "confidence": 0.0,
            }
        if module_name == "straight_keep":
            return {"omega_cmd_raw": 0.0, "omega_abs": 0.0}
        if module_name == "approach_trigger":
            return {
                "pred_label": "Straight",
                "pred_id": 0,
                "probs": {"Straight": 1.0, "NearTurnEvent": 0.0},
                "confidence": 0.0,
            }
        return {}

    def _check_model_timeout_modules(
        self, state: str, locked_turn_dir: Optional[str], now: Time
    ) -> list[str]:
        required = self._required_modules_for_state(state, locked_turn_dir)
        timed_out = []
        inflight_window_steps = max(
            1, int(self.model_output_timeout_sec * self.cmd_publish_hz)
        )
        with self._module_lock:
            module_status = {}
            for m in self._module_names:
                cache = self.module_cache[m]
                module_status[m] = {
                    "last_update_time": cache.last_update_time,
                    "busy": bool(cache.busy),
                    "future": cache.future,
                    "last_run_step": int(cache.last_run_step),
                }
        for m in required:
            status = module_status.get(m, {})
            last_t = status.get("last_update_time")
            busy = bool(status.get("busy", False))
            future = status.get("future")
            last_run_step = int(status.get("last_run_step", 0))
            age_sec = self._age_sec(last_t, now)
            step_delta = max(0, int(self._tick_count - last_run_step))
            future_running = bool(future is not None and not future.done())

            # in-flight 宽限：busy 期间允许异步任务跨若干个控制 tick 返回。
            if busy:
                # future 仍在运行中时，窗口内暂不判 timeout。
                if future_running and step_delta <= inflight_window_steps:
                    continue
                # future 已完成但结果尚未在本 tick 被回收时，窗口内同样不判 timeout。
                if (not future_running) and step_delta <= inflight_window_steps:
                    continue
                timed_out.append(m)
                continue

            if last_t is None:
                timed_out.append(m)
                continue
            if age_sec is not None and age_sec > self.model_output_timeout_sec:
                timed_out.append(m)
        return timed_out

    def _required_modules_for_state(
        self, state: str, locked_turn_dir: Optional[str]
    ) -> list[str]:
        if self.runtime_mode == "slow_straight_only":
            return ["straight_keep"]
        if state == "STRAIGHTKEEP":
            return ["straight_keep", "approach_trigger"]
        if state == "APPROACH":
            return ["stage3", "junction_lr", "straight_keep"]
        if state == "PROVISIONAL_TURN":
            return ["stage3", "junction_lr"]
        if state == "TURN":
            return ["stage3"]
        if state == "RECOVER":
            return ["straight_keep", "stage3"]
        # BOOT 默认不做严格模型超时约束，避免冷启动误触发
        return []

    # ---------------------------
    # 控制映射与安全发布
    # ---------------------------
    def _compose_control_cmd(
        self,
        state: str,
        locked_turn_dir: Optional[str],
        omega_cmd_final: float,
        straight_keep_raw_omega: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        linear_x = float(self.linear_speed_map.get(state, 0.0))
        straight_keep_trace = self._empty_straight_keep_trace()

        if state in ("STRAIGHTKEEP", "APPROACH", "RECOVER"):
            raw_omega = (
                float(straight_keep_raw_omega)
                if straight_keep_raw_omega is not None
                else float(omega_cmd_final)
            )
            angular_z, straight_keep_trace = self._calibrate_straight_keep_omega(
                raw_omega
            )
            if state == "RECOVER":
                angular_z = self._recover_same_direction_suppressed_omega(
                    omega=angular_z,
                    locked_turn_dir=locked_turn_dir,
                )
                straight_keep_trace["straight_keep_final_omega"] = float(angular_z)
        elif state == "TURN":
            if locked_turn_dir == "Left":
                angular_z = self.turn_left_omega
            elif locked_turn_dir == "Right":
                angular_z = self.turn_right_omega
            else:
                angular_z = 0.0
        elif state == "PROVISIONAL_TURN":
            angular_z = self._clip(omega_cmd_final, -0.35, 0.35)
        else:
            linear_x = float(self.linear_speed_map.get("BOOT", 0.0))
            angular_z = 0.0

        return float(linear_x), float(angular_z), straight_keep_trace

    @staticmethod
    def _clip(v: float, low: float, high: float) -> float:
        return max(low, min(high, float(v)))

    def _record_published_cmd(self, cmd: Twist) -> None:
        self._last_cmd = cmd
        self._last_angular_z = float(cmd.angular.z)
        self._last_angular_cmd_valid = abs(float(cmd.angular.z)) > 1e-9

    def _clear_last_angular_cmd(self) -> None:
        self._last_angular_z = 0.0
        self._last_angular_cmd_valid = False
        try:
            self._last_cmd.angular.z = 0.0
        except Exception:
            pass

    def publish_zero_twist(self, reason: str) -> None:
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self._record_published_cmd(cmd)
        self._clear_last_angular_cmd()
        self.get_logger().warn("Publish ZERO cmd_vel. reason=%s" % reason)

    # ---------------------------
    # 状态与调试发布
    # ---------------------------
    def _publish_state(self, state: str) -> None:
        msg = String()
        msg.data = str(state)
        self.state_pub.publish(msg)

    def _publish_debug(
        self,
        *,
        now: Time,
        state: str,
        locked_turn_dir: Optional[str],
        linear_x: float,
        angular_z: float,
        outputs: Dict[str, Dict[str, Any]],
        run_flags: Dict[str, bool],
        reason: str,
        image_age_ms: int,
        image_received_ok: Optional[bool] = None,
        sm_out: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        cmd_diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._img_lock:
            latest_image_receive_time = self._latest_image_receive_time
            latest_image_stamp = self._latest_image_stamp
            image_rx_count = int(self._image_rx_count)
            has_received_first_image = bool(self._has_received_first_image)
        with self._module_lock:
            stage3_update = self.module_cache["stage3"].last_update_time
            junction_update = self.module_cache["junction_lr"].last_update_time
            straight_keep_update = self.module_cache["straight_keep"].last_update_time
            trigger_update = self.module_cache["approach_trigger"].last_update_time
            stage3_last_run_step = int(self.module_cache["stage3"].last_run_step)
            junction_last_run_step = int(self.module_cache["junction_lr"].last_run_step)
            straight_keep_last_run_step = int(
                self.module_cache["straight_keep"].last_run_step
            )
            trigger_last_run_step = int(
                self.module_cache["approach_trigger"].last_run_step
            )
            stage3_busy = bool(self.module_cache["stage3"].busy)
            junction_busy = bool(self.module_cache["junction_lr"].busy)
            straight_keep_busy = bool(self.module_cache["straight_keep"].busy)
            trigger_busy = bool(self.module_cache["approach_trigger"].busy)
            busy_module_count = sum(
                1 for m in self._module_names if self.module_cache[m].busy
            )
            stage3_latency_ms = self.module_cache["stage3"].last_latency_ms
            junction_latency_ms = self.module_cache["junction_lr"].last_latency_ms
            straight_keep_latency_ms = self.module_cache["straight_keep"].last_latency_ms
            trigger_latency_ms = self.module_cache["approach_trigger"].last_latency_ms
            stage3_exception_count = int(self.module_cache["stage3"].exception_count)
            junction_exception_count = int(
                self.module_cache["junction_lr"].exception_count
            )
            straight_keep_exception_count = int(
                self.module_cache["straight_keep"].exception_count
            )
            trigger_exception_count = int(
                self.module_cache["approach_trigger"].exception_count
            )

        if image_received_ok is None:
            image_received_ok = (
                latest_image_receive_time is not None
                and int(image_age_ms) >= 0
                and (float(image_age_ms) / 1000.0) <= float(self.image_timeout_sec)
            )

        cmd_diag = cmd_diag if isinstance(cmd_diag, dict) else {}

        def _to_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return int(default)

        locked_turn_dir_out = (
            locked_turn_dir if locked_turn_dir in ("Left", "Right") else None
        )
        trigger_pred = self._parse_trigger_pred(outputs.get("approach_trigger", {}))
        stage3_pred = self._parse_stage3_pred(outputs.get("stage3", {}))
        junction_pred = self._parse_junction_pred(outputs.get("junction_lr", {}))
        latest_image_receive_time_str = self._format_time_stamp(latest_image_receive_time)
        image_header_stamp_str = self._format_time_stamp(latest_image_stamp)
        stage3_age_ms = self._age_ms(stage3_update, now)
        junction_age_ms = self._age_ms(junction_update, now)
        straight_keep_age_ms = self._age_ms(straight_keep_update, now)
        trigger_age_ms = self._age_ms(trigger_update, now)
        tick_count = int(self._tick_count)
        startup_elapsed_sec = self._age_sec(self._node_start_time, now)
        startup_warmup_active = (
            startup_elapsed_sec is not None
            and startup_elapsed_sec <= self.startup_warmup_sec
        )

        latest_inference_age_ms = _to_int(cmd_diag.get("latest_inference_age_ms", -1), -1)
        if latest_inference_age_ms < 0:
            valid_ages = [
                age
                for age in (stage3_age_ms, junction_age_ms, straight_keep_age_ms, trigger_age_ms)
                if isinstance(age, int) and age >= 0
            ]
            latest_inference_age_ms = min(valid_ages) if valid_ages else -1

        cmd_linear_x = self._safe_float(cmd_diag.get("cmd_linear_x", linear_x), linear_x)
        cmd_angular_z = self._safe_float(cmd_diag.get("cmd_angular_z", angular_z), angular_z)
        cmd_from_new_inference = bool(cmd_diag.get("cmd_from_new_inference", False))
        cmd_from_cached_output = bool(cmd_diag.get("cmd_from_cached_output", False))
        reused_last_cmd = bool(cmd_diag.get("reused_last_cmd", False))
        omega_source = cmd_diag.get("omega_source")
        if not isinstance(omega_source, str) or not omega_source:
            omega_source = self._resolve_omega_source(
                state=state,
                reason=reason,
                image_age_ms=int(image_age_ms),
                latest_inference_age_ms=int(latest_inference_age_ms),
                cmd_from_cached_output=cmd_from_cached_output,
            )

        omega_same_sign_as_last_turn = cmd_diag.get("omega_same_sign_as_last_turn")
        if not isinstance(omega_same_sign_as_last_turn, bool):
            omega_same_sign_as_last_turn = None

        nav_state = str(cmd_diag.get("nav_state", state))
        prev_nav_state = cmd_diag.get("prev_nav_state", None)
        state_step_count = _to_int(cmd_diag.get("state_step_count", -1), -1)
        turn_step_count = _to_int(cmd_diag.get("turn_step_count", -1), -1)
        recover_step_count = _to_int(cmd_diag.get("recover_step_count", -1), -1)
        locked_turn_dir_diag = cmd_diag.get("locked_turn_dir", locked_turn_dir_out)
        if locked_turn_dir_diag not in ("Left", "Right"):
            locked_turn_dir_diag = None
        required_modules = self._required_modules_for_state(
            state, locked_turn_dir_diag
        )
        last_turn_dir = cmd_diag.get("last_turn_dir", None)
        if last_turn_dir not in ("Left", "Right"):
            last_turn_dir = None
        last_turn_sign = cmd_diag.get("last_turn_sign", None)
        if last_turn_sign not in (-1, 1):
            last_turn_sign = None

        straight_keep_raw_omega = cmd_diag.get("straight_keep_raw_omega", None)
        straight_keep_after_bias = cmd_diag.get("straight_keep_after_bias", None)
        straight_keep_after_scale = cmd_diag.get("straight_keep_after_scale", None)
        straight_keep_after_deadband = cmd_diag.get("straight_keep_after_deadband", None)
        straight_keep_after_clip = cmd_diag.get("straight_keep_after_clip", None)
        straight_keep_final_omega = cmd_diag.get("straight_keep_final_omega", None)
        straight_keep_bias_dbg = self._safe_float(
            cmd_diag.get("straight_keep_bias", self.straight_keep_bias),
            self.straight_keep_bias,
        )
        straight_keep_scale_dbg = self._safe_float(
            cmd_diag.get("straight_keep_scale", self.straight_keep_scale),
            self.straight_keep_scale,
        )
        straight_keep_deadband_dbg = self._safe_float(
            cmd_diag.get("straight_keep_deadband", self.straight_keep_deadband),
            self.straight_keep_deadband,
        )
        model_output_timeout = self._safe_float(
            cmd_diag.get("model_output_timeout", self.model_output_timeout_sec),
            self.model_output_timeout_sec,
        )
        image_timeout = self._safe_float(
            cmd_diag.get("image_timeout", self.image_timeout_sec),
            self.image_timeout_sec,
        )
        stale_omega_suppressed = bool(cmd_diag.get("stale_omega_suppressed", False))
        stale_omega_before = cmd_diag.get("stale_omega_before", None)
        stale_omega_after = cmd_diag.get("stale_omega_after", None)
        stale_linear_hold_active = bool(
            cmd_diag.get("stale_linear_hold_active", False)
        )
        stale_linear_hold_allowed = bool(
            cmd_diag.get("stale_linear_hold_allowed", False)
        )
        stale_linear_hold_reason = str(cmd_diag.get("stale_linear_hold_reason", ""))
        max_omega_hold_sec = self._safe_float(
            cmd_diag.get("max_omega_hold_sec", self.max_omega_hold_sec),
            self.max_omega_hold_sec,
        )
        omega_stale_decay = self._safe_float(
            cmd_diag.get("omega_stale_decay", self.omega_stale_decay),
            self.omega_stale_decay,
        )
        linear_hold_speed_on_stale = self._safe_float(
            cmd_diag.get(
                "linear_hold_speed_on_stale", self.linear_hold_speed_on_stale
            ),
            self.linear_hold_speed_on_stale,
        )
        max_linear_hold_sec = self._safe_float(
            cmd_diag.get("max_linear_hold_sec", self.max_linear_hold_sec),
            self.max_linear_hold_sec,
        )
        linear_hold_max_image_age_sec = self._safe_float(
            cmd_diag.get(
                "linear_hold_max_image_age_sec", self.linear_hold_max_image_age_sec
            ),
            self.linear_hold_max_image_age_sec,
        )
        pulse_recenter_enable = bool(
            cmd_diag.get("pulse_recenter_enable", self.pulse_recenter_enable)
        )
        pulse_recenter_state = str(
            cmd_diag.get("pulse_recenter_state", self._pulse_recenter_state)
        )
        pulse_recenter_error = self._safe_float(
            cmd_diag.get(
                "pulse_recenter_error",
                self._safe_float(straight_keep_raw_omega, 0.0)
                + float(self.straight_keep_bias),
            ),
            0.0,
        )
        pulse_recenter_dir = _to_int(
            cmd_diag.get("pulse_recenter_dir", self._pulse_recenter_dir), 0
        )
        pulse_recenter_step_count = _to_int(
            cmd_diag.get(
                "pulse_recenter_step_count", self._pulse_recenter_step_count
            ),
            0,
        )
        pulse_recenter_enter_count = _to_int(
            cmd_diag.get(
                "pulse_recenter_enter_count", self._pulse_recenter_enter_count
            ),
            0,
        )
        pulse_recenter_exit_count = _to_int(
            cmd_diag.get(
                "pulse_recenter_exit_count", self._pulse_recenter_exit_count
            ),
            0,
        )
        pulse_recenter_cooldown_count = _to_int(
            cmd_diag.get(
                "pulse_recenter_cooldown_count",
                self._pulse_recenter_cooldown_count,
            ),
            0,
        )
        pulse_recenter_enter_abs = self._safe_float(
            cmd_diag.get("pulse_recenter_enter_abs", self.pulse_recenter_enter_abs),
            self.pulse_recenter_enter_abs,
        )
        pulse_recenter_exit_abs = self._safe_float(
            cmd_diag.get("pulse_recenter_exit_abs", self.pulse_recenter_exit_abs),
            self.pulse_recenter_exit_abs,
        )
        pulse_recenter_omega = self._safe_float(
            cmd_diag.get("pulse_recenter_omega", self.pulse_recenter_omega),
            self.pulse_recenter_omega,
        )
        pulse_recenter_reason = str(
            cmd_diag.get("pulse_recenter_reason", "")
        )
        active_scheduled_module = self._active_scheduled_module_this_tick
        if active_scheduled_module not in self._module_names:
            active_scheduled_module = None
        slow_safe_waiting_first_model_output = bool(
            cmd_diag.get("slow_safe_waiting_first_model_output", False)
        )
        reason_s = str(reason or "")
        safety_level_default = "NORMAL"
        hard_stop_reason_default = ""
        if reason_s == "waiting_first_straight_keep_output":
            safety_level_default = "WAITING_FIRST_MODEL"
        elif reason_s.startswith("model_output_timeout") or reason_s in (
            "missing_image",
            "missing_image_or_timeout",
            "model_inference_exception",
            "model_submit_exception",
            "too_many_consecutive_errors",
            "exception",
            "control_tick_exception",
        ):
            safety_level_default = "HARD_STOP"
            hard_stop_reason_default = reason_s
        if bool(cmd_diag.get("stale_linear_hold_active", False)):
            safety_level_default = "STALE_MODEL_CREEP"
        safety_level = str(cmd_diag.get("safety_level", safety_level_default))
        hard_stop_reason = str(
            cmd_diag.get("hard_stop_reason", hard_stop_reason_default)
        )

        debug_payload: Dict[str, Any] = {
            "runtime_build_tag": str(self.runtime_build_tag),
            "runtime_file": str(self.runtime_file),
            "config_path": str(self.config_path),
            "cmd_vel_topic": str(self.cmd_vel_topic),
            "debug_schema_version": 2,
            "infer_max_workers": int(self.infer_max_workers),
            "runtime_mode": str(self.runtime_mode),
            "single_active_model": bool(self.single_active_model),
            "busy_module_count": int(busy_module_count),
            "global_max_inflight_models": int(self.global_max_inflight_models),
            "max_submits_per_tick": int(self.max_submits_per_tick),
            "active_scheduled_module_this_tick": active_scheduled_module,
            "slow_safe_waiting_first_model_output": bool(
                slow_safe_waiting_first_model_output
            ),
            "safety_level": safety_level,
            "hard_stop_reason": hard_stop_reason,
            "state": state,
            "nav_state": nav_state,
            "prev_nav_state": prev_nav_state,
            "state_step_count": int(state_step_count),
            "turn_step_count": int(turn_step_count),
            "recover_step_count": int(recover_step_count),
            "locked_turn_dir": locked_turn_dir_diag,
            "last_turn_dir": last_turn_dir,
            "last_turn_sign": last_turn_sign,
            "linear_x": float(linear_x),
            "angular_z": float(angular_z),
            "cmd_linear_x": float(cmd_linear_x),
            "cmd_angular_z": float(cmd_angular_z),
            "omega_source": str(omega_source),
            "cmd_from_new_inference": bool(cmd_from_new_inference),
            "cmd_from_cached_output": bool(cmd_from_cached_output),
            "reused_last_cmd": bool(reused_last_cmd),
            "omega_same_sign_as_last_turn": omega_same_sign_as_last_turn,
            "trigger_pred": trigger_pred,
            "stage3_pred": stage3_pred,
            "junction_pred": junction_pred,
            "image_received_ok": bool(image_received_ok),
            "image_age_ms": int(image_age_ms),
            "latest_image_receive_time": latest_image_receive_time_str,
            "stage3_age_ms": stage3_age_ms,
            "junction_age_ms": junction_age_ms,
            "straight_keep_age_ms": straight_keep_age_ms,
            "trigger_age_ms": trigger_age_ms,
            "latest_inference_age_ms": int(latest_inference_age_ms),
            "stage3_busy": stage3_busy,
            "junction_busy": junction_busy,
            "straight_keep_busy": straight_keep_busy,
            "trigger_busy": trigger_busy,
            "stage3_latency_ms": stage3_latency_ms,
            "junction_latency_ms": junction_latency_ms,
            "straight_keep_latency_ms": straight_keep_latency_ms,
            "trigger_latency_ms": trigger_latency_ms,
            "stage3_exception_count": int(stage3_exception_count),
            "junction_exception_count": int(junction_exception_count),
            "straight_keep_exception_count": int(straight_keep_exception_count),
            "trigger_exception_count": int(trigger_exception_count),
            "ran_stage3": bool(run_flags.get("stage3", False)),
            "ran_junction": bool(run_flags.get("junction_lr", False)),
            "ran_straight_keep": bool(run_flags.get("straight_keep", False)),
            "ran_trigger": bool(run_flags.get("approach_trigger", False)),
            "straight_keep_stride": int(self.straight_keep_stride),
            "trigger_stride": int(self.trigger_stride),
            "stage3_probe_stride": int(self.stage3_probe_stride),
            "required_modules": list(required_modules),
            "tick_count": tick_count,
            "image_rx_count": image_rx_count,
            "has_received_first_image": has_received_first_image,
            "startup_warmup_active": bool(startup_warmup_active),
            "startup_warmup_sec": float(self.startup_warmup_sec),
            "straight_keep_raw_omega": straight_keep_raw_omega,
            "straight_keep_bias": float(straight_keep_bias_dbg),
            "straight_keep_scale": float(straight_keep_scale_dbg),
            "straight_keep_deadband": float(straight_keep_deadband_dbg),
            "straight_keep_after_bias": straight_keep_after_bias,
            "straight_keep_after_scale": straight_keep_after_scale,
            "straight_keep_after_deadband": straight_keep_after_deadband,
            "straight_keep_after_clip": straight_keep_after_clip,
            "straight_keep_final_omega": straight_keep_final_omega,
            "stale_omega_suppressed": bool(stale_omega_suppressed),
            "stale_omega_before": stale_omega_before,
            "stale_omega_after": stale_omega_after,
            "stale_linear_hold_active": bool(stale_linear_hold_active),
            "stale_linear_hold_allowed": bool(stale_linear_hold_allowed),
            "stale_linear_hold_reason": stale_linear_hold_reason,
            "max_omega_hold_sec": float(max_omega_hold_sec),
            "omega_stale_decay": float(omega_stale_decay),
            "linear_hold_speed_on_stale": float(linear_hold_speed_on_stale),
            "max_linear_hold_sec": float(max_linear_hold_sec),
            "linear_hold_max_image_age_sec": float(linear_hold_max_image_age_sec),
            "pulse_recenter_enable": bool(pulse_recenter_enable),
            "pulse_recenter_state": str(pulse_recenter_state),
            "pulse_recenter_error": float(pulse_recenter_error),
            "pulse_recenter_dir": int(pulse_recenter_dir),
            "pulse_recenter_step_count": int(pulse_recenter_step_count),
            "pulse_recenter_enter_count": int(pulse_recenter_enter_count),
            "pulse_recenter_exit_count": int(pulse_recenter_exit_count),
            "pulse_recenter_cooldown_count": int(
                pulse_recenter_cooldown_count
            ),
            "pulse_recenter_enter_abs": float(pulse_recenter_enter_abs),
            "pulse_recenter_exit_abs": float(pulse_recenter_exit_abs),
            "pulse_recenter_omega": float(pulse_recenter_omega),
            "pulse_recenter_reason": str(pulse_recenter_reason),
            "stage3_last_run_step": stage3_last_run_step,
            "junction_last_run_step": junction_last_run_step,
            "straight_keep_last_run_step": straight_keep_last_run_step,
            "trigger_last_run_step": trigger_last_run_step,
            "model_output_timeout": float(model_output_timeout),
            "image_timeout": float(image_timeout),
            "reason": reason,
            "consecutive_errors": int(self._consecutive_errors),
        }

        if not self.debug_compact:
            debug_payload.update(
                {
                    "subscribed_image_topic": self.image_topic,
                    "image_header_stamp": image_header_stamp_str,
                }
            )
            if sm_out is not None:
                debug_payload["state_machine_debug"] = sm_out.get("debug", {})

        if extra:
            debug_payload.update(extra)

        msg = String()
        msg.data = json.dumps(debug_payload, ensure_ascii=False)
        self.debug_pub.publish(msg)

    def _get_state_name(self) -> str:
        state_obj = getattr(self.state_machine, "state", None)
        if state_obj is None:
            return "BOOT"
        # Enum.name / str 兼容
        if hasattr(state_obj, "name"):
            return str(state_obj.name)
        return str(state_obj)

    def _get_locked_turn_dir(self) -> Optional[str]:
        v = getattr(self.state_machine, "locked_turn_dir", None)
        if v in ("Left", "Right"):
            return v
        return None

    def _get_latest_image_copy(self) -> Optional[np.ndarray]:
        with self._img_lock:
            if self._latest_image is None:
                return None
            return self._latest_image.copy()

    def _get_latest_image_status(self, now: Time) -> Tuple[bool, int]:
        with self._img_lock:
            last_receive_time = self._latest_image_receive_time or self._latest_image_time
        age_ms = self._age_ms(last_receive_time, now)
        if last_receive_time is None:
            return False, -1
        age_sec = age_ms / 1000.0
        if age_sec > self.image_timeout_sec:
            return False, age_ms
        return True, age_ms

    def _maybe_log_missing_image(self, now: Time, image_age_ms: int) -> None:
        if self._last_missing_image_log_time is not None:
            elapsed = self._age_sec(self._last_missing_image_log_time, now)
            if elapsed is not None and elapsed < self._missing_image_log_interval_sec:
                return

        with self._img_lock:
            last_receive_time = self._latest_image_receive_time or self._latest_image_time
            last_header_stamp = self._latest_image_stamp or self._latest_image_header_time

        self.get_logger().warn(
            "Image missing/timeout. topic=%s, image_age_ms=%d, last_receive_time=%s, "
            "last_header_stamp=%s"
            % (
                self.image_topic,
                int(image_age_ms),
                self._format_time_stamp(last_receive_time),
                self._format_time_stamp(last_header_stamp),
            )
        )
        self._last_missing_image_log_time = now

    @staticmethod
    def _format_time_stamp(time_obj: Optional[Time]) -> str:
        if time_obj is None:
            return ""
        ns = int(time_obj.nanoseconds)
        sec = ns // 1_000_000_000
        nsec = ns % 1_000_000_000
        return "%d.%09d" % (sec, nsec)

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _age_sec(last_time: Optional[Time], now: Time) -> Optional[float]:
        if last_time is None:
            return None
        delta_ns = (now - last_time).nanoseconds
        if delta_ns < 0:
            return 0.0
        return float(delta_ns) / 1e9

    @staticmethod
    def _age_ms(last_time: Optional[Time], now: Time) -> int:
        age = HierarchicalNavRuntimeNode._age_sec(last_time, now)
        if age is None:
            return -1
        return int(age * 1000.0)

    @staticmethod
    def _parse_stage3_pred(out: Dict[str, Any]) -> str:
        if not isinstance(out, dict):
            return ""
        name = out.get("pred_stage", "")
        if isinstance(name, str) and name:
            return name
        pid = out.get("pred_id", None)
        try:
            return {0: "Approach", 1: "Turn", 2: "Recover"}.get(int(pid), "")
        except Exception:
            return ""

    @staticmethod
    def _parse_junction_pred(out: Dict[str, Any]) -> str:
        if not isinstance(out, dict):
            return ""
        name = out.get("pred_label", "")
        if isinstance(name, str) and name:
            return name
        pid = out.get("pred_id", None)
        try:
            return {0: "Left", 1: "Right"}.get(int(pid), "")
        except Exception:
            return ""

    @staticmethod
    def _parse_trigger_pred(out: Dict[str, Any]) -> str:
        if not isinstance(out, dict):
            return ""
        name = out.get("pred_label", "")
        if isinstance(name, str) and name:
            return name
        pid = out.get("pred_id", None)
        try:
            return {0: "Straight", 1: "NearTurnEvent"}.get(int(pid), "")
        except Exception:
            return ""

    def _shutdown_inference_executor(self) -> None:
        if self._executor_shutdown:
            return

        with self._module_lock:
            for module_name in self._module_names:
                cache = self.module_cache[module_name]
                if cache.future is not None and not cache.future.done():
                    cache.future.cancel()
                cache.future = None
                cache.busy = False

        self._infer_executor.shutdown(wait=False, cancel_futures=True)
        self._executor_shutdown = True

    def destroy_node(self) -> bool:
        # 节点退出时必须发布一次零速
        try:
            self.publish_zero_twist("node_destroy")
        except Exception:
            pass
        try:
            self._shutdown_inference_executor()
        except Exception as exc:
            self.get_logger().warn("关闭推理线程池失败: %s" % str(exc))
        return super().destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[HierarchicalNavRuntimeNode] = None
    executor: Optional[MultiThreadedExecutor] = None
    try:
        node = HierarchicalNavRuntimeNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error("节点启动/运行失败: %s" % str(exc))
        else:
            print("HierarchicalNavRuntimeNode failed before init: %s" % str(exc))
    finally:
        if executor is not None:
            try:
                executor.shutdown()
            except Exception:
                pass
        if node is not None:
            try:
                node.publish_zero_twist("node_shutdown")
            except Exception:
                pass
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
