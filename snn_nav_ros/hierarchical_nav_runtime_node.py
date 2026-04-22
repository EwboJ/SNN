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


class HierarchicalNavRuntimeNode(Node):
    """真实小车层级导航在线运行时节点。"""

    def __init__(self) -> None:
        super().__init__("hierarchical_nav_runtime")

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
            1, int(self.scheduler_cfg.get("stage3_probe_stride", 3))
        )
        self.turn_stage3_stride = max(
            1, int(self.scheduler_cfg.get("turn_stage3_stride", 2))
        )
        self.recover_stage3_stride = max(
            1, int(self.scheduler_cfg.get("recover_stage3_stride", 2))
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
            max_workers=len(self._module_names),
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
        self.control_timer = self.create_timer(
            1.0 / self.cmd_publish_hz,
            self._control_tick,
            callback_group=self._control_cb_group,
        )

        if self.scheduler_policy != "state_conditioned_v2":
            self.get_logger().warn(
                "scheduler.policy=%s，当前节点仅实现 state_conditioned_v2，已按该策略运行。"
                % self.scheduler_policy
            )

        self.get_logger().info(
            "HierarchicalNavRuntimeNode started. config=%s, image_topic=%s, cmd_vel_topic=%s, "
            "state_topic=%s, debug_topic=%s, hz=%.2f"
            % (
                self.config_path,
                self.image_topic,
                self.cmd_vel_topic,
                self.state_topic,
                self.debug_topic,
                self.cmd_publish_hz,
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
        now = self.get_clock().now()

        # 当前状态（调度用）
        state_now = self._get_state_name()
        locked_now = self._get_locked_turn_dir()

        # 默认 debug 信息（即便 early-return 也可发布）
        run_flags = {
            "stage3": False,
            "junction_lr": False,
            "straight_keep": False,
            "approach_trigger": False,
        }
        outputs = self._get_cached_outputs()
        startup_elapsed_sec = self._age_sec(self._node_start_time, now)
        startup_warmup_active = (
            startup_elapsed_sec is not None
            and startup_elapsed_sec <= self.startup_warmup_sec
        )

        try:
            # 0) 先收割已完成的异步推理结果（非阻塞）
            tick_had_exception = self._collect_finished_inference_results(now=now)
            outputs = self._get_cached_outputs()
            if tick_had_exception:
                self._consecutive_errors += 1
            else:
                self._consecutive_errors = 0

            # 1) 启动 warmup 窗口：首帧未到时不触发 image timeout 停车
            if startup_warmup_active:
                with self._img_lock:
                    has_received_first_image = bool(self._has_received_first_image)
                if not has_received_first_image:
                    reason = "startup_warmup_waiting_first_image"
                    self.publish_zero_twist(reason)
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
                        image_age_ms=-1,
                        image_received_ok=False,
                    )
                    return

            # 2) 图像超时检查
            image_ok, image_age_ms = self._get_latest_image_status(now)
            if not image_ok:
                with self._img_lock:
                    has_received_first_image = bool(self._has_received_first_image)

                if not has_received_first_image:
                    reason = "waiting_first_image"
                    self.publish_zero_twist(reason)
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
                    )
                    return

                reason = "missing_image_or_timeout"
                if not startup_warmup_active:
                    self._maybe_log_missing_image(now=now, image_age_ms=image_age_ms)
                self.publish_zero_twist(reason)
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
                )
                return

            image_np = self._get_latest_image_copy()
            if image_np is None:
                reason = "missing_image"
                self.publish_zero_twist(reason)
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
                )
                return

            # 2) 按当前状态决定本轮模型调度，提交后台推理（不等待）
            schedule_flags = self._decide_schedule(state_now, locked_now)
            run_flags = self._submit_inference_jobs(
                image_np=image_np,
                schedule_flags=schedule_flags,
            )
            outputs = self._get_cached_outputs()

            # 3) 连续异常安全机制
            if self._consecutive_errors >= self.max_consecutive_errors:
                reason = "too_many_consecutive_errors"
                self.get_logger().error(
                    "连续异常达到阈值: %d >= %d，执行安全停车。"
                    % (self._consecutive_errors, self.max_consecutive_errors)
                )
                self.publish_zero_twist(reason)
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
                )
                return

            # 4) 模型输出超时检查（按当前状态依赖）
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
                self.get_logger().error("模型输出超时: %s" % ",".join(timeout_modules))
                self.publish_zero_twist(reason)
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
                )
                return

            # 5) 状态机更新
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

            # 6) 速度映射 + 角速度规则
            linear_x, angular_z = self._compose_control_cmd(
                state=state_new,
                locked_turn_dir=locked_new,
                omega_cmd_final=omega_sm,
            )

            # 7) 发布 cmd/state/debug
            cmd = Twist()
            cmd.linear.x = float(linear_x)
            cmd.angular.z = float(angular_z)
            self.cmd_pub.publish(cmd)
            self._last_cmd = cmd

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
            )
        except Exception as exc:
            self._consecutive_errors += 1
            self.get_logger().error(
                "control_tick 异常: %s\n%s" % (str(exc), traceback.format_exc())
            )
            self.publish_zero_twist("control_tick_exception")
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
                image_age_ms=self._age_ms(
                    self._latest_image_receive_time or self._latest_image_time, now
                ),
                image_received_ok=None,
                extra={"exception": str(exc)},
            )

    # ---------------------------
    # 调度 / 模型缓存
    # ---------------------------
    def _is_stride_tick(self, stride: int) -> bool:
        s = max(1, int(stride))
        # 第一个 tick 即命中（tick=1 -> 0 % s == 0）
        return ((self._tick_count - 1) % s) == 0

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

        if state == "STRAIGHTKEEP":
            run["straight_keep"] = True
            run["approach_trigger"] = True
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
                run["junction_lr"] = True
        elif state == "RECOVER":
            run["straight_keep"] = True
            run["stage3"] = self._is_stride_tick(self.recover_stage3_stride)
        else:  # BOOT/未知状态：保守预热
            run["straight_keep"] = True
            run["approach_trigger"] = True
            run["stage3"] = self._is_stride_tick(self.stage3_probe_stride)

        return run

    def _collect_finished_inference_results(self, now: Time) -> bool:
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
        for module_name, future in completed_jobs:
            try:
                out = future.result()
                if not isinstance(out, dict):
                    raise TypeError("%s 输出不是 dict" % module_name)
                with self._module_lock:
                    cache = self.module_cache[module_name]
                    cache.last_output = out
                    cache.last_update_time = now
                    cache.busy = False
                    cache.future = None
            except Exception as exc:
                had_exception = True
                with self._module_lock:
                    cache = self.module_cache[module_name]
                    cache.busy = False
                    cache.future = None
                self.get_logger().error(
                    "异步模型运行异常[%s]: %s\n%s"
                    % (module_name, str(exc), traceback.format_exc())
                )

        return had_exception

    def _submit_inference_jobs(
        self, image_np: np.ndarray, schedule_flags: Dict[str, bool]
    ) -> Dict[str, bool]:
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

        for module_name in self._module_names:
            if not schedule_flags.get(module_name, False):
                continue
            model = self.models.get(module_name)
            if model is None:
                continue

            should_submit = False
            with self._module_lock:
                cache = self.module_cache[module_name]
                if not cache.busy:
                    cache.busy = True
                    cache.last_run_step = self._tick_count
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
                with self._module_lock:
                    cache = self.module_cache[module_name]
                    cache.busy = False
                    cache.future = None
                self.get_logger().error(
                    "提交异步推理失败[%s]: %s" % (module_name, str(exc))
                )
                continue

            with self._module_lock:
                cache = self.module_cache[module_name]
                cache.future = future
            ran_flags[module_name] = True

        return ran_flags

    def _get_cached_outputs(self) -> Dict[str, Dict[str, Any]]:
        outputs: Dict[str, Dict[str, Any]] = {}
        with self._module_lock:
            for module_name in self._module_names:
                out = self.module_cache[module_name].last_output
                if not isinstance(out, dict):
                    out = self._default_output(module_name)
                outputs[module_name] = out
        return outputs

    def _run_model(self, module_name: str, image_np: np.ndarray) -> Dict[str, Any]:
        model = self.models.get(module_name)
        if model is None:
            raise RuntimeError("模型未加载: %s" % module_name)

        # 优先走 predict()
        if hasattr(model, "predict"):
            return model.predict(image_np)
        if callable(model):
            return model(image_np)
        raise RuntimeError("模型对象不可调用: %s" % module_name)

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
        if state == "STRAIGHTKEEP":
            return ["straight_keep", "approach_trigger", "stage3"]
        if state == "APPROACH":
            return ["stage3", "junction_lr", "straight_keep"]
        if state == "PROVISIONAL_TURN":
            return ["stage3", "junction_lr", "straight_keep"]
        if state == "TURN":
            req = ["stage3"]
            if not (self.disable_junction_after_lock and locked_turn_dir in ("Left", "Right")):
                req.append("junction_lr")
            return req
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
    ) -> Tuple[float, float]:
        linear_x = float(self.linear_speed_map.get(state, 0.0))

        if state in ("STRAIGHTKEEP", "APPROACH", "RECOVER"):
            angular_z = self._clip(omega_cmd_final, -self.angular_clip, self.angular_clip)
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

        return float(linear_x), float(angular_z)

    @staticmethod
    def _clip(v: float, low: float, high: float) -> float:
        return max(low, min(high, float(v)))

    def publish_zero_twist(self, reason: str) -> None:
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        self._last_cmd = cmd
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

        if image_received_ok is None:
            image_received_ok = (
                latest_image_receive_time is not None
                and int(image_age_ms) >= 0
                and (float(image_age_ms) / 1000.0) <= float(self.image_timeout_sec)
            )

        locked_turn_dir_out = locked_turn_dir if locked_turn_dir is not None else ""
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

        if self.debug_compact:
            # 轻量模式：仅发布核心诊断字段，降低 JSON 序列化与发布开销。
            debug_payload: Dict[str, Any] = {
                "state": state,
                "locked_turn_dir": locked_turn_dir_out,
                "linear_x": float(linear_x),
                "angular_z": float(angular_z),
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
                "stage3_busy": stage3_busy,
                "junction_busy": junction_busy,
                "straight_keep_busy": straight_keep_busy,
                "trigger_busy": trigger_busy,
                "tick_count": tick_count,
                "image_rx_count": image_rx_count,
                "has_received_first_image": has_received_first_image,
                "startup_warmup_active": bool(startup_warmup_active),
                "startup_warmup_sec": float(self.startup_warmup_sec),
                "stage3_last_run_step": stage3_last_run_step,
                "junction_last_run_step": junction_last_run_step,
                "straight_keep_last_run_step": straight_keep_last_run_step,
                "trigger_last_run_step": trigger_last_run_step,
                "reason": reason,
                "consecutive_errors": int(self._consecutive_errors),
            }
        else:
            debug_payload = {
                "state": state,
                "locked_turn_dir": locked_turn_dir_out,
                "linear_x": float(linear_x),
                "angular_z": float(angular_z),
                "trigger_pred": trigger_pred,
                "stage3_pred": stage3_pred,
                "junction_pred": junction_pred,
                "ran_stage3": bool(run_flags.get("stage3", False)),
                "ran_junction": bool(run_flags.get("junction_lr", False)),
                "ran_straight_keep": bool(run_flags.get("straight_keep", False)),
                "ran_trigger": bool(run_flags.get("approach_trigger", False)),
                "stage3_busy": stage3_busy,
                "junction_busy": junction_busy,
                "straight_keep_busy": straight_keep_busy,
                "trigger_busy": trigger_busy,
                "subscribed_image_topic": self.image_topic,
                "image_received_ok": bool(image_received_ok),
                "image_age_ms": int(image_age_ms),
                "image_header_stamp": image_header_stamp_str,
                "latest_image_receive_time": latest_image_receive_time_str,
                "stage3_age_ms": stage3_age_ms,
                "junction_age_ms": junction_age_ms,
                "straight_keep_age_ms": straight_keep_age_ms,
                "trigger_age_ms": trigger_age_ms,
                "tick_count": tick_count,
                "image_rx_count": image_rx_count,
                "has_received_first_image": has_received_first_image,
                "startup_warmup_active": bool(startup_warmup_active),
                "startup_warmup_sec": float(self.startup_warmup_sec),
                "stage3_last_run_step": stage3_last_run_step,
                "junction_last_run_step": junction_last_run_step,
                "straight_keep_last_run_step": straight_keep_last_run_step,
                "trigger_last_run_step": trigger_last_run_step,
                "reason": reason,
                "consecutive_errors": int(self._consecutive_errors),
            }
            if sm_out is not None:
                debug_payload["state_machine_debug"] = sm_out.get("debug", {})
            if extra:
                debug_payload.update(extra)

        msg = String()
        msg.data = json.dumps(debug_payload, ensure_ascii=False)
        self.debug_pub.publish(msg)

    # ---------------------------
    # 小工具
    # ---------------------------
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
