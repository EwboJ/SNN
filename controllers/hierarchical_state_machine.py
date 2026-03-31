"""
走廊导航层级状态机（与 ROS2 解耦，可离线/在线共用）
=====================================================

本模块只负责状态逻辑与角速度决策，不依赖 ROS2。
可被 replay 脚本和后续在线节点共同调用。

核心状态：
  - BOOT
  - STRAIGHTKEEP
  - APPROACH
  - TURN
  - RECOVER

输入 update() 的标准格式：
{
  "stage3": {
    "pred_stage": "...",
    "confidence": 0.0,
    "probs": {"Approach": ..., "Turn": ..., "Recover": ...}
  },
  "junction_lr": {
    "pred_label": "...",
    "confidence": 0.0,
    "probs": {"Left": ..., "Right": ...}
  },
  "straight_keep": {
    "omega_cmd_raw": 0.0
  }
}

示例：
    sm = HierarchicalNavigatorStateMachine(
        stage_window_size=7,
        stage_enter_turn_votes=5,
        stage_exit_turn_votes=5,
        junction_window_size=5,
        junction_lock_votes=4,
        recover_min_steps=8,
        boot_steps=6,
        straightkeep_suppress_in_turn=True,
        recover_blend_steps=12,
        max_turn_steps=20,
        use_fixed_turn_rate=True,
        omega_clip=1.2,
        use_clip=True,
    )
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Optional


class NavState(Enum):
    """导航状态枚举。"""
    BOOT = 'BOOT'
    STRAIGHTKEEP = 'STRAIGHTKEEP'
    APPROACH = 'APPROACH'
    TURN = 'TURN'
    RECOVER = 'RECOVER'


class HierarchicalNavigatorStateMachine:
    """
    走廊层级导航状态机。

    设计要点：
    1) 所有状态转移都使用窗口投票或连续计数，避免逐帧抖动。
    2) junction 方向仅在进入 TURN 时锁存，TURN 期间不可修改。
    3) RECOVER 通过线性混合逐步恢复 straight_keep 控制权。
    """

    def __init__(
        self,
        stage_window_size: int = 7,
        stage_enter_turn_votes: int = 5,
        stage_exit_turn_votes: int = 5,
        junction_window_size: int = 5,
        junction_lock_votes: int = 4,
        recover_min_steps: int = 8,
        boot_steps: int = 6,
        straightkeep_suppress_in_turn: bool = True,
        recover_blend_steps: int = 12,
        max_turn_steps: int = 20,
        use_fixed_turn_rate: bool = True,
        omega_clip: float = 1.2,
        use_clip: bool = True,
        # 可选控制参数（未在强制列表中，但便于工程接入）
        left_turn_omega: float = 1.2,
        right_turn_omega: float = -1.2,
        # TURN -> RECOVER 支持累计门槛（渐减模式下需要的有效支持步数）
        recover_support_steps_needed: int = 2,
        # ===== 新增可配置参数 =====
        # TURN 最小步数：在此之前不允许退出 TURN
        min_turn_steps: int = 5,
        # TURN 退出投票阈值：turn_votes 低于此值时考虑退出
        turn_exit_vote_threshold: int = 2,
        # TURN 退出时 straight_keep omega 阈值：|omega| 低于此值视为可安全切回直行
        straight_recover_omega_thresh: float = 0.3,
        # TURN 退出时低 turn+低 omega 需持续的步数
        straight_recover_hold_steps: int = 3,
        # APPROACH 中 junction 最早锁存的步数门槛
        min_approach_steps_before_junction_lock: int = 3,
        # APPROACH 中 junction 锁存需要的最低 turn 票数
        min_turn_votes_before_junction_lock: int = 2,
        start_junction_hist_on_turn_signal: bool = True,
        min_turn_votes_to_start_junction_hist: int = 1,
        reset_junction_hist_when_no_turn_signal: bool = True,
    ) -> None:
        # 参数做安全裁剪，避免异常配置导致状态机不可用
        self.stage_window_size = max(1, int(stage_window_size))
        self.stage_enter_turn_votes = max(1, int(stage_enter_turn_votes))
        self.stage_exit_turn_votes = max(1, int(stage_exit_turn_votes))
        self.junction_window_size = max(1, int(junction_window_size))
        self.junction_lock_votes = max(1, int(junction_lock_votes))
        self.recover_min_steps = max(1, int(recover_min_steps))
        self.boot_steps = max(0, int(boot_steps))
        self.straightkeep_suppress_in_turn = bool(straightkeep_suppress_in_turn)
        self.recover_blend_steps = max(1, int(recover_blend_steps))
        # 与 hierarchical_nav.yaml 对齐的系统级控制参数
        self.max_turn_steps = max(1, int(max_turn_steps))
        self.use_fixed_turn_rate = bool(use_fixed_turn_rate)
        self.omega_clip = max(0.0, float(omega_clip))
        self.use_clip = bool(use_clip)

        self.left_turn_omega = float(left_turn_omega)
        self.right_turn_omega = float(right_turn_omega)
        self.recover_support_steps_needed = max(1, int(recover_support_steps_needed))

        # ===== 新增参数 =====
        self.min_turn_steps = max(1, int(min_turn_steps))
        self.turn_exit_vote_threshold = max(0, int(turn_exit_vote_threshold))
        self.straight_recover_omega_thresh = max(0.0, float(straight_recover_omega_thresh))
        self.straight_recover_hold_steps = max(1, int(straight_recover_hold_steps))
        self.min_approach_steps_before_junction_lock = max(0, int(min_approach_steps_before_junction_lock))
        self.min_turn_votes_before_junction_lock = max(0, int(min_turn_votes_before_junction_lock))
        self.start_junction_hist_on_turn_signal = bool(start_junction_hist_on_turn_signal)
        self.min_turn_votes_to_start_junction_hist = max(0, int(min_turn_votes_to_start_junction_hist))
        self.reset_junction_hist_when_no_turn_signal = bool(reset_junction_hist_when_no_turn_signal)

        # 历史窗口（用于投票迟滞）
        self._stage_hist = deque(maxlen=self.stage_window_size)
        self._junction_hist = deque(maxlen=self.junction_window_size)

        self.reset()

    def reset(self) -> None:
        """
        重置状态机。
        建议每个 run 开始前调用一次。
        """
        self.state: NavState = NavState.BOOT
        self.locked_turn_dir: Optional[str] = None
        self.global_step: int = 0
        self.state_step: int = 0
        self._recover_support_count: int = 0
        # 新增：低 turn + 低 omega 持续计数
        self._straight_recover_hold_count: int = 0
        self._stage_hist.clear()
        self._junction_hist.clear()

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _normalize_stage_name(name: Any) -> Optional[str]:
        if name is None:
            return None
        s = str(name).strip().lower()
        mapping = {
            'approach': 'Approach',
            'turn': 'Turn',
            'recover': 'Recover',
        }
        return mapping.get(s, None)

    @staticmethod
    def _normalize_turn_dir(name: Any) -> Optional[str]:
        if name is None:
            return None
        s = str(name).strip().lower()
        mapping = {
            'left': 'Left',
            'l': 'Left',
            'right': 'Right',
            'r': 'Right',
        }
        return mapping.get(s, None)

    @staticmethod
    def _argmax_key(d: Any) -> Optional[str]:
        if not isinstance(d, dict) or not d:
            return None
        best_k = None
        best_v = None
        for k, v in d.items():
            try:
                vf = float(v)
            except Exception:
                continue
            if best_v is None or vf > best_v:
                best_v = vf
                best_k = str(k)
        return best_k

    def _parse_stage_pred(self, stage_out: Dict[str, Any]) -> Optional[str]:
        # 优先 pred_stage，其次 probs argmax，其次 pred_id 兜底
        stage_name = self._normalize_stage_name(stage_out.get('pred_stage'))
        if stage_name:
            return stage_name

        k = self._argmax_key(stage_out.get('probs'))
        stage_name = self._normalize_stage_name(k)
        if stage_name:
            return stage_name

        pid = stage_out.get('pred_id', None)
        id_map = {0: 'Approach', 1: 'Turn', 2: 'Recover'}
        try:
            return id_map.get(int(pid), None)
        except Exception:
            return None

    def _parse_junction_pred(self, junction_out: Dict[str, Any]) -> Optional[str]:
        # 优先 pred_label，其次 probs argmax，其次 pred_id 兜底
        turn_dir = self._normalize_turn_dir(junction_out.get('pred_label'))
        if turn_dir:
            return turn_dir

        k = self._argmax_key(junction_out.get('probs'))
        turn_dir = self._normalize_turn_dir(k)
        if turn_dir:
            return turn_dir

        pid = junction_out.get('pred_id', None)
        id_map = {0: 'Left', 1: 'Right'}
        try:
            return id_map.get(int(pid), None)
        except Exception:
            return None

    def _majority_threshold(self) -> int:
        return self.stage_window_size // 2 + 1

    def _get_stage_counts(self) -> Counter:
        return Counter(self._stage_hist)

    def _get_junction_counts(self) -> Counter:
        return Counter(self._junction_hist)

    def _get_stage_majority(self, counts: Counter) -> Optional[str]:
        # 固定优先级保证可复现（同票时不随机）
        order = ('Approach', 'Turn', 'Recover')
        best_label = None
        best_count = -1
        for lb in order:
            c = int(counts.get(lb, 0))
            if c > best_count:
                best_count = c
                best_label = lb
        if best_count >= self._majority_threshold():
            return best_label
        return None

    def _get_junction_locked_candidate(self, counts: Counter) -> Optional[str]:
        # Left/Right 任一满足锁存票数即可
        left_c = int(counts.get('Left', 0))
        right_c = int(counts.get('Right', 0))
        if left_c >= self.junction_lock_votes and left_c >= right_c:
            return 'Left'
        if right_c >= self.junction_lock_votes and right_c > left_c:
            return 'Right'
        return None

    def _is_turn_rising(self) -> bool:
        # Keep compatibility with previous behavior: latest stage is Turn => rising.
        hist_len = len(self._stage_hist)
        if hist_len >= 2:
            recent_2 = list(self._stage_hist)[-2:]
            return recent_2[-1] == 'Turn'
        if hist_len >= 1:
            recent_1 = list(self._stage_hist)[-1:]
            return recent_1[-1] == 'Turn'
        return False

    def _fixed_turn_omega(self, turn_dir: Optional[str]) -> float:
        if turn_dir == 'Left':
            return self.left_turn_omega
        if turn_dir == 'Right':
            return self.right_turn_omega
        return 0.0

    def _transition(self, to_state: NavState, reason: str) -> Dict[str, Any]:
        from_state = self.state
        self.state = to_state
        self.state_step = 0
        return {
            'from': from_state.name,
            'to': to_state.name,
            'reason': reason,
        }

    def update(self, module_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新状态机并输出当前控制结果。

        Args:
            module_outputs: 三个子模块当前帧输出。

        Returns:
            dict:
              - state
              - locked_turn_dir
              - omega_cmd_final
              - submodule_outputs
              - debug
        """
        # ===== 1) 读取输入并做安全兜底 =====
        stage_out = deepcopy(module_outputs.get('stage3', {}) or {})
        junction_out = deepcopy(module_outputs.get('junction_lr', {}) or {})
        straight_out = deepcopy(module_outputs.get('straight_keep', {}) or {})

        stage_pred = self._parse_stage_pred(stage_out)
        junction_pred = self._parse_junction_pred(junction_out)
        omega_raw = self._safe_float(straight_out.get('omega_cmd_raw', 0.0), 0.0)

        # ===== 2) 更新时间步与历史窗口 =====
        self.global_step += 1
        self.state_step += 1

        if stage_pred is not None:
            self._stage_hist.append(stage_pred)

        # 仅在 APPROACH 缓存 junction 方向，避免旧窗口污染
        stage_counts = self._get_stage_counts()
        turn_votes = int(stage_counts.get('Turn', 0))
        _turn_rising = self._is_turn_rising()
        junction_hist_update_allowed = False
        junction_hist_reset_applied = False

        if self.state == NavState.APPROACH:
            _turn_signal_for_junction_hist = (
                turn_votes >= self.min_turn_votes_to_start_junction_hist
                or stage_pred == 'Turn'
                or _turn_rising
            )
            junction_hist_update_allowed = (
                (not self.start_junction_hist_on_turn_signal)
                or _turn_signal_for_junction_hist
            )
            if junction_hist_update_allowed:
                if junction_pred is not None:
                    self._junction_hist.append(junction_pred)
            elif self.reset_junction_hist_when_no_turn_signal and turn_votes == 0:
                if self._junction_hist:
                    self._junction_hist.clear()
                    junction_hist_reset_applied = True
        elif self.state in (NavState.BOOT, NavState.STRAIGHTKEEP):
            self._junction_hist.clear()

        stage_counts = self._get_stage_counts()
        junction_counts = self._get_junction_counts()
        stage_majority = self._get_stage_majority(stage_counts)
        junction_candidate = self._get_junction_locked_candidate(junction_counts)

        approach_votes = int(stage_counts.get('Approach', 0))
        turn_votes = int(stage_counts.get('Turn', 0))
        recover_votes = int(stage_counts.get('Recover', 0))
        majority_thr = self._majority_threshold()
        enter_turn_thr = max(majority_thr, self.stage_enter_turn_votes)

        transition_info = None
        turn_timeout_triggered = False
        # 用于 debug：记录 fallback 被阻止的原因
        fallback_blocked_by_turn_signal = False
        fallback_blocked_by_junction_lock = False
        # 新增 debug 字段
        turn_exit_ready = False
        junction_lock_allowed = True
        junction_lock_block_reason = ''

        # ===== 3) 状态转移逻辑（含迟滞） =====
        if self.state == NavState.BOOT:
            # BOOT 预热完成后进入 STRAIGHTKEEP
            if self.state_step >= self.boot_steps:
                transition_info = self._transition(
                    NavState.STRAIGHTKEEP,
                    f'boot_steps_reached({self.boot_steps})'
                )

        elif self.state == NavState.STRAIGHTKEEP:
            # 进入直行时清除本次转弯锁存
            self.locked_turn_dir = None
            self._recover_support_count = 0
            # STRAIGHTKEEP -> APPROACH: stage3 Approach 多数投票
            if approach_votes >= majority_thr:
                transition_info = self._transition(
                    NavState.APPROACH,
                    f'approach_votes({approach_votes}) >= majority({majority_thr})'
                )
                # 新一次接近路口，重新累计 junction 投票
                self._junction_hist.clear()

        elif self.state == NavState.APPROACH:
            # ===== APPROACH 转移优先级：TURN 进入 > fallback =====
            # 判断 Turn 票数是否正在上升（最近两帧趋势）
            _turn_rising = self._is_turn_rising()
            _hist_len = len(self._stage_hist)
            if _hist_len >= 2:
                # 检查最近两帧中至少有一帧是 Turn
                _recent_2 = list(self._stage_hist)[-2:]
                _turn_rising = (_recent_2[-1] == 'Turn')
            elif _hist_len >= 1:
                _recent_1 = list(self._stage_hist)[-1:]
                _turn_rising = (_recent_1[-1] == 'Turn')

            # ===== Junction 锁存门控逻辑（防止早期锁错） =====
            # junction_candidate 是原始投票判定结果，这里通过额外门控决定是否允许正式锁存
            _junction_raw_candidate = junction_candidate  # 保留原始候选用于 debug
            _jlock_time_ok = (self.state_step >= self.min_approach_steps_before_junction_lock)
            _jlock_turn_ok = (turn_votes >= self.min_turn_votes_before_junction_lock)
            _jlock_rising_ok = _turn_rising

            if not _jlock_time_ok:
                # 步数不足，延迟锁存
                junction_lock_allowed = False
                junction_lock_block_reason = 'junction_lock_deferred_by_early_approach'
                junction_candidate = None  # 不允许使用，但继续累积 junction votes
            elif not (_jlock_turn_ok or _jlock_rising_ok):
                # Turn 信号太弱且无上升趋势，延迟锁存
                junction_lock_allowed = False
                junction_lock_block_reason = 'junction_lock_deferred_by_weak_turn_signal'
                junction_candidate = None
            else:
                junction_lock_allowed = True
                junction_lock_block_reason = ''

            # 计算宽松进入 TURN 的阈值
            _soft_turn_thr = max(1, self.stage_enter_turn_votes - 1)

            # ---- 条件 A: junction 已锁定 + Turn 票数 >= 多数阈值 ----
            _cond_a = (junction_candidate is not None
                       and turn_votes >= majority_thr)
            # ---- 条件 B: junction 已锁定 + Turn 票数 >= 进入阈值 ----
            _cond_b = (junction_candidate is not None
                       and turn_votes >= self.stage_enter_turn_votes)
            # ---- 条件 C: junction 已锁定 + Turn 票数 >= (进入阈值-1) 且趋势上升 ----
            _cond_c = (junction_candidate is not None
                       and turn_votes >= _soft_turn_thr
                       and _turn_rising)

            if _cond_a or _cond_b or _cond_c:
                # ===== APPROACH -> TURN =====
                self.locked_turn_dir = junction_candidate
                # 选择最精确的 reason 标签
                if _cond_b or (turn_votes >= enter_turn_thr):
                    _reason = 'normal_enter_turn'
                elif _cond_a:
                    _reason = 'enter_turn_with_locked_junction_candidate'
                else:
                    _reason = 'enter_turn_with_locked_junction_candidate'
                transition_info = self._transition(
                    NavState.TURN,
                    f'{_reason}(turn_votes={turn_votes}, '
                    f'junction_lock={junction_candidate}, '
                    f'rising={_turn_rising})'
                )
                self._recover_support_count = 0
                self._straight_recover_hold_count = 0
            else:
                # ===== APPROACH -> STRAIGHTKEEP (保守 fallback) =====
                # 必须同时满足所有条件才允许 fallback
                _fb_time_ok = (self.state_step >= self.stage_window_size)
                _fb_approach_weak = (approach_votes < majority_thr)
                _fb_turn_absent = (turn_votes < max(1, self.stage_enter_turn_votes - 1))
                _fb_no_junction = (_junction_raw_candidate is None)

                if _fb_time_ok and _fb_approach_weak and _fb_turn_absent and _fb_no_junction:
                    transition_info = self._transition(
                        NavState.STRAIGHTKEEP,
                        f'approach_fallback_no_turn_signal('
                        f'approach={approach_votes}, turn={turn_votes}, '
                        f'junction={_junction_raw_candidate})'
                    )
                    self._junction_hist.clear()
                else:
                    # 记录 fallback 被阻止的原因（用于 debug 分析）
                    if _fb_time_ok and _fb_approach_weak:
                        if not _fb_turn_absent:
                            fallback_blocked_by_turn_signal = True
                        if not _fb_no_junction:
                            fallback_blocked_by_junction_lock = True

        elif self.state == NavState.TURN:
            # TURN 期间方向锁存不可改写（关键约束）
            # ===== Recover 支持累计逻辑（渐减模式，吸收抖动） =====
            if recover_votes >= majority_thr:
                self._recover_support_count += 1
            else:
                # 不立刻清零，渐减以吸收 Recover 的轻微抖动
                self._recover_support_count = max(0, self._recover_support_count - 1)

            # ===== 低 turn + 低 omega 持续计数 =====
            _low_turn = (turn_votes <= self.turn_exit_vote_threshold)
            _low_omega = (abs(omega_raw) <= self.straight_recover_omega_thresh)
            if _low_turn and _low_omega:
                self._straight_recover_hold_count += 1
            else:
                self._straight_recover_hold_count = 0

            # ===== TURN -> RECOVER 多条件退出逻辑 =====
            # 只有过了最小 TURN 步数才允许进入 RECOVER 判断
            turn_exit_ready = (self.state_step >= self.min_turn_steps)

            if turn_exit_ready:
                # 条件 a: Recover 信号已稳定确认
                _exit_by_recover_signal = (
                    self._recover_support_count >= self.recover_support_steps_needed
                )
                # 条件 b: 低 turn 票数 + 低 omega 持续足够步数
                _exit_by_low_turn_omega = (
                    self._straight_recover_hold_count >= self.straight_recover_hold_steps
                )

                if _exit_by_recover_signal:
                    transition_info = self._transition(
                        NavState.RECOVER,
                        f'recover_signal_confirmed('
                        f'support={self._recover_support_count}, '
                        f'needed={self.recover_support_steps_needed})'
                    )
                elif _exit_by_low_turn_omega:
                    transition_info = self._transition(
                        NavState.RECOVER,
                        f'recover_by_low_turn_and_low_omega('
                        f'turn_votes={turn_votes}, '
                        f'omega_raw={omega_raw:.3f}, '
                        f'hold_count={self._straight_recover_hold_count})'
                    )

            # 最终兜底：timeout 强制退出
            if transition_info is None and self.state_step >= self.max_turn_steps:
                turn_timeout_triggered = True
                transition_info = self._transition(
                    NavState.RECOVER,
                    f'turn_timeout(max_turn_steps={self.max_turn_steps}, '
                    f'recover_support={self._recover_support_count})'
                )
                self._recover_support_count = 0
                self._straight_recover_hold_count = 0

        elif self.state == NavState.RECOVER:
            # RECOVER 保持至少 recover_min_steps，且 Turn 不再占多数后回到 STRAIGHTKEEP
            if self.state_step >= self.recover_min_steps and turn_votes < majority_thr:
                transition_info = self._transition(
                    NavState.STRAIGHTKEEP,
                    f'recover_stable(state_step={self.state_step}, turn_votes={turn_votes})'
                )
                self.locked_turn_dir = None
                self._recover_support_count = 0
                self._junction_hist.clear()

        # ===== 4) 计算最终角速度输出 =====
        turn_omega = self._fixed_turn_omega(self.locked_turn_dir)
        recover_alpha = 1.0
        turn_mix_ratio = 0.0

        if self.state == NavState.BOOT:
            omega_cmd_final = 0.0
        elif self.state in (NavState.STRAIGHTKEEP, NavState.APPROACH):
            # 直行和接近阶段，主要采用 straight_keep 输出
            omega_cmd_final = omega_raw
        elif self.state == NavState.TURN:
            # TURN 阶段按配置切换控制策略，且锁存方向始终主导
            if self.use_fixed_turn_rate:
                omega_cmd_final = turn_omega
            else:
                # 允许少量 straight_keep 混合；suppress=True 时混合更弱
                turn_mix_ratio = 0.1 if self.straightkeep_suppress_in_turn else 0.3
                omega_cmd_final = ((1.0 - turn_mix_ratio) * turn_omega
                                   + turn_mix_ratio * omega_raw)
                # 强约束：转向方向以锁存方向为主，防止反向穿越
                if turn_omega != 0.0 and (omega_cmd_final * turn_omega) < 0.0:
                    sign = 1.0 if turn_omega > 0.0 else -1.0
                    omega_cmd_final = sign * abs(omega_cmd_final)
        elif self.state == NavState.RECOVER:
            # RECOVER 阶段线性混合: 从 turn 控制逐步过渡到 straight_keep
            recover_alpha = min(1.0, self.state_step / float(self.recover_blend_steps))
            omega_cmd_final = (1.0 - recover_alpha) * turn_omega + recover_alpha * omega_raw
        else:
            omega_cmd_final = omega_raw

        # 统一裁剪：在最终角速度上执行系统级安全限制
        omega_before_clip = float(omega_cmd_final)
        clip_applied = False
        if self.use_clip:
            low = -abs(self.omega_clip)
            high = abs(self.omega_clip)
            omega_cmd_final = max(low, min(high, float(omega_cmd_final)))
            clip_applied = (abs(omega_cmd_final - omega_before_clip) > 1e-12)
        omega_after_clip = float(omega_cmd_final)

        # ===== 5) 组织调试输出 =====
        debug = {
            'global_step': self.global_step,
            'state_step': self.state_step,
            'stage_pred': stage_pred,
            'junction_pred': junction_pred,
            'stage_votes': {
                'Approach': approach_votes,
                'Turn': turn_votes,
                'Recover': recover_votes,
            },
            'junction_votes': {
                'Left': int(junction_counts.get('Left', 0)),
                'Right': int(junction_counts.get('Right', 0)),
            },
            'stage_majority': stage_majority,
            'junction_candidate': junction_candidate,
            'recover_support_count': self._recover_support_count,
            'recover_support_threshold': self.recover_support_steps_needed,
            'thresholds': {
                'stage_majority': majority_thr,
                'stage_enter_turn_votes': enter_turn_thr,
                'stage_exit_turn_votes': self.stage_exit_turn_votes,
                'junction_lock_votes': self.junction_lock_votes,
                'recover_min_steps': self.recover_min_steps,
                'max_turn_steps': self.max_turn_steps,
                'recover_support_steps_needed': self.recover_support_steps_needed,
                'min_turn_steps': self.min_turn_steps,
                'turn_exit_vote_threshold': self.turn_exit_vote_threshold,
                'straight_recover_omega_thresh': self.straight_recover_omega_thresh,
                'straight_recover_hold_steps': self.straight_recover_hold_steps,
                'min_approach_steps_before_junction_lock': self.min_approach_steps_before_junction_lock,
                'min_turn_votes_before_junction_lock': self.min_turn_votes_before_junction_lock,
                'start_junction_hist_on_turn_signal': self.start_junction_hist_on_turn_signal,
                'min_turn_votes_to_start_junction_hist': self.min_turn_votes_to_start_junction_hist,
                'reset_junction_hist_when_no_turn_signal': self.reset_junction_hist_when_no_turn_signal,
            },
            'omega_components': {
                'straight_keep_raw': float(omega_raw),
                'turn_component': float(turn_omega),
                'recover_alpha': float(recover_alpha),
                'turn_mix_ratio': float(turn_mix_ratio),
                'use_fixed_turn_rate': bool(self.use_fixed_turn_rate),
            },
            'turn_timeout_triggered': bool(turn_timeout_triggered),
            # Recover 信号分析
            'current_recover_votes': recover_votes,
            'clip_applied': bool(clip_applied),
            'omega_before_clip': float(omega_before_clip),
            'omega_after_clip': float(omega_after_clip),
            # fallback 被阻止的原因分析
            'fallback_blocked_by_turn_signal': bool(fallback_blocked_by_turn_signal),
            'fallback_blocked_by_junction_lock': bool(fallback_blocked_by_junction_lock),
            # ===== 新增 debug 字段 =====
            'turn_exit_ready': bool(turn_exit_ready),
            'straight_recover_hold_count': int(self._straight_recover_hold_count),
            'junction_lock_allowed': bool(junction_lock_allowed),
            'junction_lock_block_reason': str(junction_lock_block_reason),
            'junction_hist_update_allowed': bool(junction_hist_update_allowed),
            'junction_hist_reset_applied': bool(junction_hist_reset_applied),
            'transition': transition_info,
        }

        return {
            'state': self.state.name,
            'locked_turn_dir': self.locked_turn_dir,
            'omega_cmd_final': float(omega_cmd_final),
            'submodule_outputs': {
                'stage3': stage_out,
                'junction_lr': junction_out,
                'straight_keep': straight_out,
            },
            'debug': debug,
        }


__all__ = ['NavState', 'HierarchicalNavigatorStateMachine']
