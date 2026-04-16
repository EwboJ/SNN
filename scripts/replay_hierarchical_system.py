"""
层级导航系统离线回放脚本（系统级研究分析版，不依赖 ROS2）
=========================================================

功能：
1. 读取层级导航配置（configs/hierarchical_nav.yaml）
2. 按调度策略调用四模块推理（stage3 / junction_lr / straight_keep / approach_trigger）
3. 调用层级状态机 update() 得到系统级控制输出
4. 输出回放轨迹、汇总统计和时间轴图

输入 run 目录要求：
  - 必须包含 images/
  - 可选 labels.csv（若存在，优先按其 image_name 顺序回放）
  - 可选 meta.json

输出文件：
  - replay_trace.csv
  - replay_summary.json
  - state_timeline.png
  - replay_debug.json（当 logging.save_debug_json=true）

示例命令：
  # 常规回放（读取 yaml 中模型路径与状态机参数）
  python scripts/replay_hierarchical_system.py ^
      --run_dir data/corridor/test/J1_left_r02 ^
      --config configs/hierarchical_nav.yaml ^
      --out_dir results/replay_J1_left_r02 ^
      --device cuda:0

  # 快速调试前 120 帧
  python scripts/replay_hierarchical_system.py ^
      --run_dir data/corridor/test/J1_left_r02 ^
      --max_steps 120

  # 仅回放 labels.csv 中 valid=1 的帧
  python scripts/replay_hierarchical_system.py ^
      --run_dir data/corridor/test/J1_left_r02 ^
      --valid_only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('缺少依赖 PyYAML，请先安装: pip install pyyaml') from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None



# 允许脚本在仓库根目录直接运行
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.corridor_module_infer import (  # noqa: E402
    JunctionLRInfer,
    Stage3Infer,
    StraightKeepInfer,
    ApproachTriggerInfer,
)
from controllers.hierarchical_state_machine import (  # noqa: E402
    HierarchicalNavigatorStateMachine,
)


def _natural_key(text: str) -> List[Any]:
    """文件名自然排序 key（例如 2 < 10）。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', text)]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _percentile(values: List[float], q: float) -> float:
    """计算百分位数（线性插值），空列表返回 0.0。"""
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    qq = max(0.0, min(100.0, float(q)))
    rank = (qq / 100.0) * (len(arr) - 1)
    low = int(rank)
    high = min(low + 1, len(arr) - 1)
    frac = rank - low
    return arr[low] * (1.0 - frac) + arr[high] * frac


def _summarize_timing_ms(values: List[float]) -> Dict[str, float]:
    """汇总耗时序列，统一输出 avg/p50/p90/p95/max。"""
    if not values:
        return {
            'avg': 0.0,
            'p50': 0.0,
            'p90': 0.0,
            'p95': 0.0,
            'max': 0.0,
            'count': 0.0,
        }
    arr = [float(v) for v in values]
    return {
        'avg': sum(arr) / len(arr),
        'p50': _percentile(arr, 50.0),
        'p90': _percentile(arr, 90.0),
        'p95': _percentile(arr, 95.0),
        'max': max(arr),
        'count': float(len(arr)),
    }


def _truncate_tail(text: Any, max_len: int = 28) -> str:
    """截断过长文本，优先保留尾部（便于看文件名后缀）。"""
    s = str(text)
    if max_len <= 0:
        return ''
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[-max_len:]
    return '...' + s[-(max_len - 3):]


def _dedupe_paths(paths: List[str]) -> List[str]:
    """Preserve order while removing duplicate absolute paths."""
    out: List[str] = []
    seen = set()
    for p in paths:
        key = os.path.normcase(os.path.normpath(p))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _resolve_path_with_candidates(
    path_str: str,
    base_dir: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    Unified path resolution:
    1) absolute path (after expanding ~ and env vars)
    2) relative path resolved in order: repo_root -> base_dir -> cwd
    Returns:
        (best_effort_resolved_path, tried_absolute_candidates)
    """
    raw = str(path_str or '').strip()
    if not raw:
        return '', []

    expanded = os.path.expandvars(os.path.expanduser(raw))

    if os.path.isabs(expanded):
        candidates = [os.path.abspath(expanded)]
    else:
        candidates = [os.path.abspath(os.path.join(_REPO_ROOT, expanded))]
        if base_dir:
            candidates.append(os.path.abspath(os.path.join(base_dir, expanded)))
        candidates.append(os.path.abspath(os.path.join(os.getcwd(), expanded)))
        candidates = _dedupe_paths(candidates)

    for p in candidates:
        if os.path.exists(p):
            return p, candidates
    return candidates[0], candidates


def _resolve_path(path_str: str, base_dir: Optional[str] = None) -> str:
    resolved, _ = _resolve_path_with_candidates(path_str, base_dir=base_dir)
    return resolved


def _format_missing_path_message(raw_path: str, tried_candidates: List[str]) -> str:
    lines = [
        f'raw_path={raw_path}',
        'tried_candidates:',
    ]
    if tried_candidates:
        lines.extend(f'  - {p}' for p in tried_candidates)
    else:
        lines.append('  - <none>')
    return '\n'.join(lines)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'配置文件格式错误（顶层必须是 dict）: {path}')
    return data


def _load_optional_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _load_labels_rows(labels_csv: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not os.path.isfile(labels_csv):
        return [], []
    rows: List[Dict[str, Any]] = []
    with open(labels_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))
    return rows, fields


def _is_valid_flag(val: Any) -> bool:
    """判断 labels.csv 中 valid 字段值是否表示有效帧。"""
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ('1', 'true', 'yes')


def _collect_frames(
    run_dir: str,
    valid_only: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str], int, int]:
    """
    收集回放帧：
    1) labels.csv 存在且包含 image_name 时，按 labels 顺序回放；
    2) 否则按 images/ 文件名自然排序回放。

    当 valid_only=True 且 labels.csv 含 valid 字段时，仅保留有效帧。

    Returns:
        frames: 帧列表
        label_fields: labels.csv 列名
        original_count: 过滤前总帧数
        skipped_count: 被 valid 过滤掉的帧数
    """
    images_dir = os.path.join(run_dir, 'images')
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f'run 目录缺少 images/: {run_dir}')

    labels_csv = os.path.join(run_dir, 'labels.csv')
    label_rows, label_fields = _load_labels_rows(labels_csv)

    frames: List[Dict[str, Any]] = []
    original_count = 0
    skipped_count = 0

    # 检测 labels.csv 是否含有 valid 字段
    has_valid_field = bool(label_rows and ('valid' in label_rows[0]))
    do_valid_filter = valid_only and has_valid_field

    if label_rows and ('image_name' in label_rows[0]):
        for row in label_rows:
            image_name = str(row.get('image_name', '')).strip()
            if not image_name:
                continue
            image_path = os.path.join(images_dir, image_name)
            if not os.path.isfile(image_path):
                continue
            original_count += 1
            # valid 过滤
            if do_valid_filter and not _is_valid_flag(row.get('valid')):
                skipped_count += 1
                continue
            frames.append({
                'image_name': image_name,
                'image_path': image_path,
                'label_row': row,
            })
    else:
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        names = [n for n in os.listdir(images_dir)
                 if os.path.splitext(n)[1].lower() in valid_ext]
        names = sorted(names, key=_natural_key)
        for n in names:
            original_count += 1
            frames.append({
                'image_name': n,
                'image_path': os.path.join(images_dir, n),
                'label_row': {},
            })

    if not frames:
        raise RuntimeError(f'没有可用图像帧: {run_dir}')
    return frames, label_fields, original_count, skipped_count


def _parse_timestamp_ns(val: Any) -> Optional[int]:
    """安全解析 timestamp_ns，失败返回 None。"""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        try:
            return int(float(s))
        except Exception:
            return None


def _can_use_time_sampling(frames: List[Dict[str, Any]]) -> bool:
    """判断是否可对当前帧序列使用 timestamp_ns 时间采样。"""
    if not frames:
        return False
    has_timestamp_field = False
    has_parseable_ts = False
    for fr in frames:
        row = fr.get('label_row', {})
        if not isinstance(row, dict):
            continue
        if 'timestamp_ns' in row:
            has_timestamp_field = True
            if _parse_timestamp_ns(row.get('timestamp_ns')) is not None:
                has_parseable_ts = True
    return has_timestamp_field and has_parseable_ts


def _apply_frame_stride(
    frames: List[Dict[str, Any]],
    frame_stride: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """按固定步长抽帧。"""
    stride = max(1, int(frame_stride))
    if stride <= 1:
        return frames, 0
    sampled = [fr for i, fr in enumerate(frames) if (i % stride) == 0]
    skipped = len(frames) - len(sampled)
    return sampled, skipped


def _apply_time_sampling(
    frames: List[Dict[str, Any]],
    sample_dt_ms: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """按 timestamp_ns 做时间间隔采样。"""
    dt_ms = float(sample_dt_ms)
    if dt_ms <= 0.0:
        return frames, 0

    min_dt_ns = dt_ms * 1e6
    sampled: List[Dict[str, Any]] = []
    last_keep_ts: Optional[int] = None

    for fr in frames:
        row = fr.get('label_row', {})
        ts = _parse_timestamp_ns(row.get('timestamp_ns') if isinstance(row, dict) else None)
        if ts is None:
            # 缺失时间戳时跳过，避免破坏时间间隔采样约束
            continue
        if last_keep_ts is None:
            sampled.append(fr)
            last_keep_ts = ts
            continue
        if (ts - last_keep_ts) >= min_dt_ns:
            sampled.append(fr)
            last_keep_ts = ts

    skipped = len(frames) - len(sampled)
    return sampled, skipped


def _apply_sampling(
    frames: List[Dict[str, Any]],
    frame_stride: int,
    sample_dt_ms: float,
) -> Tuple[List[Dict[str, Any]], int, str]:
    """
    统一采样入口：
    1) 若 sample_dt_ms > 0 且 timestamp_ns 可用，优先时间采样
    2) 否则按 frame_stride 采样
    返回: (采样后帧, 采样跳过帧数, 采样模式)
    """
    if float(sample_dt_ms) > 0.0 and _can_use_time_sampling(frames):
        sampled, skipped = _apply_time_sampling(frames, sample_dt_ms)
        return sampled, skipped, 'time'

    sampled, skipped = _apply_frame_stride(frames, frame_stride)
    mode = 'stride' if max(1, int(frame_stride)) > 1 else 'none'
    return sampled, skipped, mode


def _normalize_phase(phase: str) -> str:
    """将 phase 文本规整到 Approach/Turn/Recover；无法识别返回空串。"""
    s = str(phase or '').strip().lower()
    if s == 'approach':
        return 'Approach'
    if s == 'turn':
        return 'Turn'
    if s == 'recover':
        return 'Recover'
    return ''


def _infer_gt_turn_dir(run_name: str) -> str:
    """
    从 run_name 推断真实转向方向：
      *_left_*  -> Left
      *_right_* -> Right
    """
    s = str(run_name or '').strip().lower()
    if re.search(r'(^|_)left(_|$)', s):
        return 'Left'
    if re.search(r'(^|_)right(_|$)', s):
        return 'Right'
    return ''


def _compress_state_sequence(states: List[str]) -> List[str]:
    """将状态序列压缩为首次变化序列，用于论文展示状态链路。"""
    seq: List[str] = []
    prev = None
    for st in states:
        if st != prev:
            seq.append(st)
            prev = st
    return seq


def _first_step_with_state(trace_rows: List[Dict[str, Any]], target_state: str) -> Optional[int]:
    for r in trace_rows:
        if str(r.get('state', '')) == target_state:
            return int(r['step_idx'])
    return None


def _extract_state_segments(
    trace_rows: List[Dict[str, Any]],
    target_state: str,
) -> List[List[int]]:
    """
    提取 target_state 的连续段，返回 [[start_step, end_step], ...]。
    """
    segments: List[List[int]] = []
    start_step: Optional[int] = None
    prev_step: Optional[int] = None

    for r in trace_rows:
        step = int(_safe_float(r.get('step_idx', 0), 0))
        st = str(r.get('state', ''))
        if st == target_state:
            if start_step is None:
                start_step = step
            prev_step = step
        else:
            if start_step is not None and prev_step is not None:
                segments.append([int(start_step), int(prev_step)])
                start_step = None
                prev_step = None

    if start_step is not None and prev_step is not None:
        segments.append([int(start_step), int(prev_step)])
    return segments


def _parse_turn_exit_reason(reason: Any) -> str:
    """将 TURN->RECOVER 的 transition_reason 归一为主退出原因标签。"""
    reason_lc = str(reason or '').lower()
    if 'recover_signal_confirmed' in reason_lc:
        return 'recover_signal'
    if 'soft_exit' in reason_lc:
        return 'soft_exit'
    if 'recover_by_low_turn_and_low_omega' in reason_lc:
        return 'low_turn_low_omega'
    if 'turn_timeout' in reason_lc:
        return 'timeout'
    return 'none'


def _get_turn_exit_reason_primary(trace_rows: List[Dict[str, Any]]) -> str:
    """
    从 trace_rows 中提取主 TURN 退出原因：
    - 取首个 TURN -> RECOVER 转移的 reason 进行解析
    - 若不存在该转移，则返回 none
    """
    for r in trace_rows:
        from_state = str(r.get('transition_from', ''))
        to_state = str(r.get('transition_to', ''))
        if from_state == 'TURN' and to_state == 'RECOVER':
            return _parse_turn_exit_reason(r.get('transition_reason', ''))
    return 'none'


def _parse_bool_arg(val: Any) -> bool:
    """解析布尔命令行参数，支持 true/false/1/0/yes/no。"""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if s in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'无法解析布尔值: {val}')


def _safe_stride(v: Any, default: int = 1) -> int:
    try:
        vv = int(v)
    except Exception:
        vv = int(default)
    return max(1, vv)


def _should_run_by_stride(step_idx: int, stride: int) -> bool:
    s = max(1, int(stride))
    return (int(step_idx) % s) == 0


def _default_module_outputs() -> Dict[str, Dict[str, Any]]:
    """四个子模块的安全默认输出。"""
    return {
        'stage3': {
            'pred_stage': '',
            'pred_id': -1,
            'probs': {},
            'confidence': 0.0,
        },
        'junction_lr': {
            'pred_label': '',
            'pred_id': -1,
            'probs': {},
            'confidence': 0.0,
        },
        'straight_keep': {
            'omega_cmd_raw': 0.0,
            'omega_abs': 0.0,
        },
        'approach_trigger': {
            'pred_label': '',
            'pred_id': -1,
            'probs': {},
            'confidence': 0.0,
        },
    }


def _resolve_module_output_from_cache(
    module_name: str,
    module_cache: Dict[str, Dict[str, Any]],
    module_defaults: Dict[str, Dict[str, Any]],
    reuse_last_outputs: bool,
) -> Dict[str, Any]:
    """
    当某模块本步未运行时：
    - reuse_last_outputs=True 且缓存存在：返回 last_output
    - 否则：返回安全默认输出
    """
    cache = module_cache.get(module_name, {}) or {}
    last_output = cache.get('last_output', None)
    if reuse_last_outputs and isinstance(last_output, dict):
        return deepcopy(last_output)
    return deepcopy(module_defaults[module_name])


def _build_module_run_plan(
    scheduler_policy: str,
    state_name: str,
    step_idx: int,
    locked_turn_dir: Optional[str],
    stage3_probe_stride: int,
    junction_probe_stride: int,
    straight_keep_stride: int,
    trigger_stride: int,
    has_trigger_model: bool,
) -> Dict[str, bool]:
    """
    生成本步模块调度计划（True=运行模型，False=复用缓存/默认值）。

    policy:
      - all_models: 与历史逻辑一致，按步全量运行（trigger 仅在模型存在时运行）
      - state_conditioned_v2: 按状态多速率调度
    """
    policy = str(scheduler_policy or 'all_models').strip().lower()
    state = str(state_name or '').strip().upper()
    has_lock = bool(str(locked_turn_dir or '').strip())

    if policy == 'all_models':
        return {
            'stage3': True,
            'junction_lr': True,
            'straight_keep': True,
            'approach_trigger': bool(has_trigger_model),
        }

    # state_conditioned_v2
    plan = {
        'stage3': False,
        'junction_lr': False,
        'straight_keep': False,
        'approach_trigger': False,
    }

    if state in ('BOOT', 'STRAIGHTKEEP'):
        plan['straight_keep'] = _should_run_by_stride(step_idx, straight_keep_stride)
        plan['approach_trigger'] = bool(has_trigger_model) and _should_run_by_stride(step_idx, trigger_stride)
        plan['stage3'] = _should_run_by_stride(step_idx, stage3_probe_stride)
        plan['junction_lr'] = False
    elif state == 'APPROACH':
        plan['stage3'] = True
        plan['junction_lr'] = _should_run_by_stride(step_idx, junction_probe_stride)
        plan['straight_keep'] = _should_run_by_stride(step_idx, straight_keep_stride)
        plan['approach_trigger'] = False
    elif state == 'PROVISIONAL_TURN':
        plan['stage3'] = True
        plan['junction_lr'] = _should_run_by_stride(step_idx, junction_probe_stride)
        plan['straight_keep'] = False
        plan['approach_trigger'] = False
    elif state == 'TURN':
        plan['stage3'] = _should_run_by_stride(step_idx, 2)
        # TURN 内在方向已锁存后停止 junction 推理
        plan['junction_lr'] = (not has_lock) and _should_run_by_stride(step_idx, junction_probe_stride)
        plan['straight_keep'] = False
        plan['approach_trigger'] = False
    elif state == 'RECOVER':
        plan['stage3'] = _should_run_by_stride(step_idx, 2)
        plan['junction_lr'] = False
        plan['straight_keep'] = _should_run_by_stride(step_idx, straight_keep_stride)
        plan['approach_trigger'] = False
    else:
        # 未知状态退化到保守探测策略
        plan['stage3'] = _should_run_by_stride(step_idx, stage3_probe_stride)
        plan['junction_lr'] = False
        plan['straight_keep'] = _should_run_by_stride(step_idx, straight_keep_stride)
        plan['approach_trigger'] = bool(has_trigger_model) and _should_run_by_stride(step_idx, trigger_stride)
    return plan


def _effective_hz(call_count: int, elapsed_sec: float) -> float:
    if elapsed_sec <= 1e-12:
        return 0.0
    return float(call_count) / float(elapsed_sec)


def _build_state_machine(cfg: Dict[str, Any]) -> HierarchicalNavigatorStateMachine:
    """根据 yaml 参数构建状态机（与配置字段完整对齐）。"""
    sm_cfg = deepcopy(cfg.get('state_machine', {}) or {})
    turn_cfg = deepcopy(cfg.get('turn_control', {}) or {})
    sk_cfg = deepcopy(cfg.get('straight_keep', {}) or {})

    return HierarchicalNavigatorStateMachine(
        stage_window_size=sm_cfg.get('stage_window_size', 7),
        stage_enter_turn_votes=sm_cfg.get('stage_enter_turn_votes', 5),
        stage_exit_turn_votes=sm_cfg.get('stage_exit_turn_votes', 5),
        junction_window_size=sm_cfg.get('junction_window_size', 5),
        junction_lock_votes=sm_cfg.get('junction_lock_votes', 4),
        recover_min_steps=sm_cfg.get('recover_min_steps', 8),
        boot_steps=sm_cfg.get('boot_steps', 6),
        straightkeep_suppress_in_turn=sm_cfg.get('straightkeep_suppress_in_turn', True),
        recover_blend_steps=sm_cfg.get('recover_blend_steps', 12),
        # 关键增强参数（来自 hierarchical_nav.yaml）
        max_turn_steps=turn_cfg.get('max_turn_steps', 20),
        use_fixed_turn_rate=turn_cfg.get('use_fixed_turn_rate', True),
        omega_clip=sk_cfg.get('omega_clip', 1.2),
        use_clip=sk_cfg.get('use_clip', True),
        # 保留原有转向角速度参数
        left_turn_omega=turn_cfg.get('left_omega', 1.2),
        right_turn_omega=turn_cfg.get('right_omega', -1.2),
        # TURN -> RECOVER 渐减累计门槛
        recover_support_steps_needed=sm_cfg.get('recover_support_steps_needed', 2),
        min_turn_steps=sm_cfg.get('min_turn_steps', 5),
        turn_exit_vote_threshold=sm_cfg.get('turn_exit_vote_threshold', 2),
        straight_recover_omega_thresh=sm_cfg.get('straight_recover_omega_thresh', 0.3),
        straight_recover_hold_steps=sm_cfg.get('straight_recover_hold_steps', 3),
        min_approach_steps_before_junction_lock=sm_cfg.get('min_approach_steps_before_junction_lock', 3),
        min_turn_votes_before_junction_lock=sm_cfg.get('min_turn_votes_before_junction_lock', 2),
        start_junction_hist_on_turn_signal=sm_cfg.get('start_junction_hist_on_turn_signal', True),
        min_turn_votes_to_start_junction_hist=sm_cfg.get('min_turn_votes_to_start_junction_hist', 1),
        reset_junction_hist_when_no_turn_signal=sm_cfg.get('reset_junction_hist_when_no_turn_signal', True),
        turn_relock_window_steps=sm_cfg.get('turn_relock_window_steps', 12),
        turn_relock_votes=sm_cfg.get('turn_relock_votes', 3),
        turn_relock_hist_size=sm_cfg.get('turn_relock_hist_size', 5),
        allow_turn_relock_once=sm_cfg.get('allow_turn_relock_once', True),
        soft_turn_steps=sm_cfg.get('soft_turn_steps', 14),
        soft_exit_min_turn_steps=sm_cfg.get('soft_exit_min_turn_steps', 12),
        soft_exit_recover_votes_needed=sm_cfg.get('soft_exit_recover_votes_needed', 1),
        soft_exit_low_omega_thresh=sm_cfg.get('soft_exit_low_omega_thresh', 0.18),
        soft_exit_turn_scale_end=sm_cfg.get('soft_exit_turn_scale_end', 0.55),
        provisional_turn_window_steps=sm_cfg.get('provisional_turn_window_steps', 6),
        provisional_turn_commit_votes=sm_cfg.get('provisional_turn_commit_votes', 4),
        provisional_turn_mix_ratio=sm_cfg.get('provisional_turn_mix_ratio', 0.35),
        provisional_turn_min_observe_steps=sm_cfg.get('provisional_turn_min_observe_steps', 3),
        provisional_turn_recent_consistency_steps=sm_cfg.get('provisional_turn_recent_consistency_steps', 3),
        provisional_turn_margin_votes=sm_cfg.get('provisional_turn_margin_votes', 2),
        provisional_turn_use_omega_gate=sm_cfg.get('provisional_turn_use_omega_gate', True),
        provisional_turn_omega_sign_thresh=sm_cfg.get('provisional_turn_omega_sign_thresh', 0.06),
        provisional_turn_require_omega_agreement=sm_cfg.get(
            'provisional_turn_require_omega_agreement', True
        ),
    )


def _snapshot_state_machine_thresholds(
    sm: HierarchicalNavigatorStateMachine,
) -> Dict[str, Any]:
    """提取状态机阈值快照，用于写入 replay_summary.json。"""
    majority_thr = int(getattr(sm, 'stage_window_size', 7)) // 2 + 1
    enter_turn_thr = max(majority_thr, int(getattr(sm, 'stage_enter_turn_votes', 5)))
    return {
        'stage_majority': majority_thr,
        'stage_enter_turn_votes': enter_turn_thr,
        'stage_exit_turn_votes': int(getattr(sm, 'stage_exit_turn_votes', 5)),
        'junction_lock_votes': int(getattr(sm, 'junction_lock_votes', 4)),
        'recover_min_steps': int(getattr(sm, 'recover_min_steps', 8)),
        'max_turn_steps': int(getattr(sm, 'max_turn_steps', 20)),
        'recover_support_steps_needed': int(getattr(sm, 'recover_support_steps_needed', 2)),
        'min_turn_steps': int(getattr(sm, 'min_turn_steps', 5)),
        'turn_exit_vote_threshold': int(getattr(sm, 'turn_exit_vote_threshold', 2)),
        'straight_recover_omega_thresh': float(getattr(sm, 'straight_recover_omega_thresh', 0.3)),
        'straight_recover_hold_steps': int(getattr(sm, 'straight_recover_hold_steps', 3)),
        'min_approach_steps_before_junction_lock': int(
            getattr(sm, 'min_approach_steps_before_junction_lock', 3)
        ),
        'min_turn_votes_before_junction_lock': int(
            getattr(sm, 'min_turn_votes_before_junction_lock', 2)
        ),
        'start_junction_hist_on_turn_signal': bool(
            getattr(sm, 'start_junction_hist_on_turn_signal', True)
        ),
        'min_turn_votes_to_start_junction_hist': int(
            getattr(sm, 'min_turn_votes_to_start_junction_hist', 1)
        ),
        'reset_junction_hist_when_no_turn_signal': bool(
            getattr(sm, 'reset_junction_hist_when_no_turn_signal', True)
        ),
        'turn_relock_window_steps': int(getattr(sm, 'turn_relock_window_steps', 12)),
        'turn_relock_votes': int(getattr(sm, 'turn_relock_votes', 3)),
        'turn_relock_hist_size': int(getattr(sm, 'turn_relock_hist_size', 5)),
        'allow_turn_relock_once': bool(getattr(sm, 'allow_turn_relock_once', True)),
        'soft_turn_steps': int(getattr(sm, 'soft_turn_steps', 14)),
        'soft_exit_min_turn_steps': int(getattr(sm, 'soft_exit_min_turn_steps', 12)),
        'soft_exit_recover_votes_needed': int(getattr(sm, 'soft_exit_recover_votes_needed', 1)),
        'soft_exit_low_omega_thresh': float(getattr(sm, 'soft_exit_low_omega_thresh', 0.18)),
        'soft_exit_turn_scale_end': float(getattr(sm, 'soft_exit_turn_scale_end', 0.55)),
        'provisional_turn_window_steps': int(getattr(sm, 'provisional_turn_window_steps', 6)),
        'provisional_turn_commit_votes': int(getattr(sm, 'provisional_turn_commit_votes', 4)),
        'provisional_turn_mix_ratio': float(getattr(sm, 'provisional_turn_mix_ratio', 0.35)),
        'provisional_turn_min_observe_steps': int(
            getattr(sm, 'provisional_turn_min_observe_steps', 3)
        ),
        'provisional_turn_recent_consistency_steps': int(
            getattr(sm, 'provisional_turn_recent_consistency_steps', 3)
        ),
        'provisional_turn_margin_votes': int(getattr(sm, 'provisional_turn_margin_votes', 2)),
        'provisional_turn_use_omega_gate': bool(
            getattr(sm, 'provisional_turn_use_omega_gate', True)
        ),
        'provisional_turn_omega_sign_thresh': float(
            getattr(sm, 'provisional_turn_omega_sign_thresh', 0.06)
        ),
        'provisional_turn_require_omega_agreement': bool(
            getattr(sm, 'provisional_turn_require_omega_agreement', True)
        ),
    }


def _plot_state_timeline(trace_rows: List[Dict[str, Any]], out_png: str) -> None:
    """
    绘制 4 行时间轴图：
    1) 状态 state
    2) stage3 预测 + gt_phase（若可得）
    3) junction 预测方向 + 锁存方向
    4) omega_raw 与 omega_final
    """
    if not trace_rows:
        Image.new('RGB', (1280, 720), color=(255, 255, 255)).save(out_png)
        return

    # 延迟导入：避免环境中 matplotlib/numpy 版本冲突导致脚本启动失败
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        from PIL import ImageDraw
        canvas = Image.new('RGB', (1400, 900), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        lines = [
            'state_timeline.png (fallback)',
            '当前环境无法导入 matplotlib，已输出占位图。',
            f'错误: {e}',
            '',
            '建议安装兼容版本后重跑以生成完整曲线图。',
            '例如: pip install --upgrade matplotlib numpy',
        ]
        y = 40
        for line in lines:
            draw.text((40, y), line, fill=(20, 20, 20))
            y += 34
        canvas.save(out_png)
        return

    steps = [int(r['step_idx']) for r in trace_rows]
    state_map = {
        'BOOT': 0,
        'STRAIGHTKEEP': 1,
        'APPROACH': 2,
        'PROVISIONAL_TURN': 3,
        'TURN': 4,
        'RECOVER': 5,
    }
    stage_map = {'Approach': 0, 'Turn': 1, 'Recover': 2}
    turn_map = {'Left': 0, 'Right': 1}

    state_vals = [state_map.get(str(r.get('state', '')), -1) for r in trace_rows]
    stage_vals = [stage_map.get(str(r.get('pred_stage', '')), -1) for r in trace_rows]
    pred_turn_vals = [turn_map.get(str(r.get('pred_turn_dir', '')), -1) for r in trace_rows]
    locked_turn_vals = [turn_map.get(str(r.get('locked_turn_dir', '')), -1) for r in trace_rows]
    omega_raw_vals = [_safe_float(r.get('omega_cmd_raw', 0.0), 0.0) for r in trace_rows]
    omega_final_vals = [_safe_float(r.get('omega_cmd_final', 0.0), 0.0) for r in trace_rows]

    # gt_phase 可选覆盖
    gt_phase_vals = [stage_map.get(_normalize_phase(r.get('gt_phase', '')), -1)
                     for r in trace_rows]
    gt_valid_idx = [i for i, v in enumerate(gt_phase_vals) if v >= 0]

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    ax = axes[0]
    ax.plot(steps, state_vals, drawstyle='steps-post', lw=1.8, color='#1E88E5')
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(['BOOT', 'STRAIGHTKEEP', 'APPROACH', 'PROVISIONAL_TURN', 'TURN', 'RECOVER'])
    ax.set_ylabel('State')
    ax.set_title('Hierarchical State Timeline')
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(steps, stage_vals, drawstyle='steps-post', lw=1.6,
            color='#43A047', label='stage3_pred')
    if gt_valid_idx:
        ax.scatter([steps[i] for i in gt_valid_idx],
                   [gt_phase_vals[i] for i in gt_valid_idx],
                   s=12, marker='x', color='#E53935', alpha=0.8, label='gt_phase')
        ax.legend(loc='upper right')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Approach', 'Turn', 'Recover'])
    ax.set_ylabel('Stage3')
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(steps, pred_turn_vals, drawstyle='steps-post',
            lw=1.4, color='#FB8C00', label='junction_pred')
    ax.plot(steps, locked_turn_vals, drawstyle='steps-post',
            lw=1.8, color='#8E24AA', label='locked_turn_dir')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Left', 'Right'])
    ax.set_ylabel('Turn Dir')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.25)

    ax = axes[3]
    ax.plot(steps, omega_raw_vals, lw=1.2, color='#546E7A', label='omega_raw')
    ax.plot(steps, omega_final_vals, lw=1.8, color='#D81B60', label='omega_final')
    ax.axhline(0.0, color='black', lw=0.8, alpha=0.5)
    ax.set_ylabel('Omega')
    ax.set_xlabel('Step Index')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)


def run_replay(args: argparse.Namespace) -> None:
    config_path = _resolve_path(args.config, base_dir=_REPO_ROOT)
    cfg = _load_yaml(config_path)
    cfg_dir = os.path.dirname(config_path)

    # 基准模式与输出控制
    benchmark_only = bool(getattr(args, 'benchmark_only', False))
    benchmark_warmup = max(0, int(getattr(args, 'benchmark_warmup', 50)))
    benchmark_steps = max(1, int(getattr(args, 'benchmark_steps', 500)))
    no_save_outputs = bool(getattr(args, 'no_save_outputs', False))
    scheduler_policy = str(getattr(args, 'scheduler_policy', 'all_models') or 'all_models').strip()
    if scheduler_policy not in ('all_models', 'state_conditioned_v2'):
        raise ValueError(
            f'不支持的 --scheduler_policy={scheduler_policy}，'
            '可选: all_models / state_conditioned_v2'
        )
    stage3_probe_stride = _safe_stride(getattr(args, 'stage3_probe_stride', 3), default=3)
    junction_probe_stride = _safe_stride(getattr(args, 'junction_probe_stride', 1), default=1)
    straight_keep_stride = _safe_stride(getattr(args, 'straight_keep_stride', 1), default=1)
    trigger_stride = _safe_stride(getattr(args, 'trigger_stride', 1), default=1)
    reuse_last_outputs = bool(getattr(args, 'reuse_last_outputs', True))

    # logging 配置（真正生效）
    log_cfg = deepcopy(cfg.get('logging', {}) or {})
    save_debug_json = bool(log_cfg.get('save_debug_json', True))
    save_csv = bool(log_cfg.get('save_csv', True))
    verbose = bool(log_cfg.get('verbose', True))
    # benchmark_only 时关闭长日志，减少非推理开销干扰
    if benchmark_only:
        verbose = False

    def vprint(msg: str) -> None:
        if verbose:
            print(msg)

    run_dir = _resolve_path(args.run_dir, base_dir=_REPO_ROOT)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f'run_dir 不存在: {run_dir}')

    if args.out_dir:
        out_dir = _resolve_path(args.out_dir, base_dir=_REPO_ROOT)
    else:
        out_dir = os.path.join(_REPO_ROOT, 'results', f'replay_{os.path.basename(run_dir)}')
    os.makedirs(out_dir, exist_ok=True)

    meta_json = _load_optional_json(os.path.join(run_dir, 'meta.json'))

    # 收集帧（支持 valid_only 过滤）
    valid_only = bool(getattr(args, 'valid_only', False))
    frame_stride = max(1, int(getattr(args, 'frame_stride', 1)))
    sample_dt_ms = max(0.0, float(getattr(args, 'sample_dt_ms', 0.0)))
    frames, label_fields, original_total, skipped_invalid = _collect_frames(
        run_dir, valid_only=valid_only
    )
    after_valid_filter_total = len(frames)
    frames, skipped_by_sampling, sampling_mode = _apply_sampling(
        frames=frames,
        frame_stride=frame_stride,
        sample_dt_ms=sample_dt_ms,
    )
    after_sampling_total = len(frames)
    if args.max_steps is not None and args.max_steps > 0:
        frames = frames[:args.max_steps]
    if benchmark_only:
        frames = frames[:benchmark_steps]

    vprint('=' * 72)
    vprint('[Replay] 层级导航系统离线回放')
    vprint(f'  配置文件:   {config_path}')
    vprint(f'  run 目录:   {run_dir}')
    vprint(f'  原始帧数:   {original_total}')
    vprint(f'  valid 过滤后帧数: {after_valid_filter_total}  (跳过无效帧: {skipped_invalid})')
    vprint(f'  采样后帧数: {after_sampling_total}  (采样跳过: {skipped_by_sampling})')
    vprint(f'  sample_dt_ms: {sample_dt_ms}')
    vprint(f'  frame_stride: {frame_stride}  (sampling_mode={sampling_mode})')
    vprint(f'  实际回放帧数: {len(frames)}')
    vprint(f'  valid_only: {valid_only}')
    vprint(f'  scheduler_policy: {scheduler_policy}')
    vprint(
        f'  scheduler_stride: stage3_probe={stage3_probe_stride}, '
        f'junction_probe={junction_probe_stride}, '
        f'straight_keep={straight_keep_stride}, trigger={trigger_stride}'
    )
    vprint(f'  reuse_last_outputs: {reuse_last_outputs}')
    vprint(f'  输出目录:   {out_dir}')
    vprint(f'  no_save_outputs: {no_save_outputs}')
    if benchmark_only:
        vprint(f'  benchmark:  only=True, warmup={benchmark_warmup}, steps={benchmark_steps}')
    vprint(f'  logging:    save_debug_json={save_debug_json}, '
           f'save_csv={save_csv}, verbose={verbose}')
    vprint('=' * 72)

    # 1) 子模块推理封装（stage3 / junction_lr / straight_keep / approach_trigger）
    model_cfg = cfg.get('models', {}) or {}
    def _require_ckpt_path(ckpt_key: str) -> str:
        raw_ckpt = str(model_cfg.get(ckpt_key, '') or '').strip()
        if not raw_ckpt:
            raise ValueError(f'models.{ckpt_key} is empty.')
        resolved_ckpt, tried_candidates = _resolve_path_with_candidates(
            raw_ckpt, base_dir=cfg_dir
        )
        if not os.path.isfile(resolved_ckpt):
            raise FileNotFoundError(
                f'models.{ckpt_key} checkpoint not found.\n'
                + _format_missing_path_message(raw_ckpt, tried_candidates)
            )
        return resolved_ckpt

    stage3_ckpt = _require_ckpt_path('stage3_ckpt')
    junction_ckpt = _require_ckpt_path('junction_lr_ckpt')
    straight_ckpt = _require_ckpt_path('straight_keep_ckpt')

    stage3_infer = Stage3Infer(stage3_ckpt, device=args.device)
    junction_infer = JunctionLRInfer(junction_ckpt, device=args.device)
    straight_infer = StraightKeepInfer(straight_ckpt, device=args.device)

    # 2) 层级状态机
    sm = _build_state_machine(cfg)
    sm.reset()
    threshold_snapshot = _snapshot_state_machine_thresholds(sm)

    # 2b) 可选：加载 approach_trigger 轻量触发模型
    approach_trigger_infer = None
    _trigger_ckpt_raw = str(model_cfg.get('approach_trigger_ckpt', '') or '').strip()
    if _trigger_ckpt_raw:
        _trigger_ckpt, _trigger_candidates = _resolve_path_with_candidates(
            _trigger_ckpt_raw, base_dir=cfg_dir
        )
        if os.path.isfile(_trigger_ckpt):
            approach_trigger_infer = ApproachTriggerInfer(_trigger_ckpt, device=args.device)
            vprint(f'  approach_trigger: 已加载 ({_trigger_ckpt})')
        else:
            vprint(
                '  approach_trigger: checkpoint not found, skip.\n'
                + _format_missing_path_message(_trigger_ckpt_raw, _trigger_candidates)
            )
    else:
        vprint('  approach_trigger: 未配置，使用旧逻辑')

    module_names = ('stage3', 'junction_lr', 'straight_keep', 'approach_trigger')
    module_defaults = _default_module_outputs()
    # 四模块缓存：last_output + last_step
    module_cache: Dict[str, Dict[str, Any]] = {
        name: {
            'last_output': None,
            'last_step': None,
        }
        for name in module_names
    }
    module_call_count: Dict[str, int] = {name: 0 for name in module_names}

    # 3) 逐帧回放
    trace_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    state_counts = Counter()
    locked_turn_dir_counts = Counter()
    num_turn_entries = 0
    num_recover_entries = 0
    num_clip_applied = 0
    num_turn_timeout_exits = 0
    num_recover_signal_exits = 0
    num_low_turn_low_omega_exits = 0
    num_soft_exit_exits = 0

    # 新增：系统级研究分析追踪变量
    turn_signal_peak_votes = 0        # Turn 票数历史峰值
    junction_lock_first_step = None   # junction 首次锁定的 step
    fallback_step_list: List[int] = []  # 所有 fallback 发生的 step

    # 检测 labels.csv 中是否有 action_name / label_name 字段
    first_junction_lock_allowed_step: Optional[int] = None
    has_action_name = 'action_name' in label_fields
    has_label_name = 'label_name' in label_fields
    has_valid_field = 'valid' in label_fields

    # 新增计时变量：用于实时进度与 benchmark_only 分析
    replay_start_time = time.perf_counter()
    recent_step_times = deque(maxlen=30)
    total_frames = len(frames)

    # benchmark_only 的逐帧耗时缓存（单位 ms）
    benchmark_preprocess_ms: List[float] = []
    benchmark_stage3_ms: List[float] = []
    benchmark_junction_ms: List[float] = []
    benchmark_straight_ms: List[float] = []
    benchmark_trigger_ms: List[float] = []
    benchmark_state_machine_ms: List[float] = []
    benchmark_total_step_ms: List[float] = []
    benchmark_call_count: Dict[str, int] = {name: 0 for name in module_names}

    # 仅非 benchmark 模式启用 tqdm，避免进度条额外开销影响纯性能测量
    use_tqdm = (tqdm is not None) and (not benchmark_only)
    if use_tqdm:
        progress = tqdm(
            enumerate(frames),
            total=total_frames,
            desc='[Replay]',
            dynamic_ncols=True,
            unit='frame',
            smoothing=0.1,
            leave=True,
        )
    else:
        progress = enumerate(frames)
        if (not benchmark_only) and verbose and (tqdm is None):
            print('[Replay][Warn] 未安装 tqdm，已退化为普通循环（可执行: pip install tqdm）')

    for idx, fr in progress:
        step_t0 = time.perf_counter()
        image_name = fr['image_name']
        image_path = fr['image_path']
        label_row = fr.get('label_row', {}) or {}

        # 真实标签辅助信息（可选）
        run_name = str(label_row.get('run_name', os.path.basename(run_dir)))
        gt_phase = str(label_row.get('phase', '')).strip() if ('phase' in label_row) else ''
        gt_turn_dir = _infer_gt_turn_dir(run_name)

        # 新增：gt 辅助标签
        gt_action_name = str(label_row.get('action_name', '')).strip() if has_action_name else ''
        gt_label_name = str(label_row.get('label_name', '')).strip() if has_label_name else ''
        valid_flag = str(label_row.get('valid', '')).strip() if has_valid_field else ''

        # 分模块计时：读图+预处理、按策略调度模型推理、状态机更新
        io_t0 = time.perf_counter()
        with Image.open(image_path) as img:
            img_rgb = img.convert('RGB')
        preprocess_ms = (time.perf_counter() - io_t0) * 1000.0

        _state_obj = getattr(sm, 'state', '')
        state_before = str(getattr(_state_obj, 'name', _state_obj) or '')
        locked_turn_before = getattr(sm, 'locked_turn_dir', None)
        run_plan = _build_module_run_plan(
            scheduler_policy=scheduler_policy,
            state_name=state_before,
            step_idx=idx,
            locked_turn_dir=locked_turn_before,
            stage3_probe_stride=stage3_probe_stride,
            junction_probe_stride=junction_probe_stride,
            straight_keep_stride=straight_keep_stride,
            trigger_stride=trigger_stride,
            has_trigger_model=(approach_trigger_infer is not None),
        )
        ran_stage3 = bool(run_plan['stage3'])
        ran_junction = bool(run_plan['junction_lr'])
        ran_straight_keep = bool(run_plan['straight_keep'])
        ran_trigger = bool(run_plan['approach_trigger'])

        stage_out: Dict[str, Any]
        junction_out: Dict[str, Any]
        straight_out: Dict[str, Any]
        trigger_out: Dict[str, Any]
        stage3_ms = 0.0
        junction_ms = 0.0
        straight_ms = 0.0
        trigger_ms = 0.0

        if ran_stage3:
            t0 = time.perf_counter()
            stage_out = stage3_infer.predict(img_rgb)
            stage3_ms = (time.perf_counter() - t0) * 1000.0
            module_cache['stage3']['last_output'] = deepcopy(stage_out)
            module_cache['stage3']['last_step'] = idx
            module_call_count['stage3'] += 1
        else:
            stage_out = _resolve_module_output_from_cache(
                'stage3', module_cache, module_defaults, reuse_last_outputs
            )

        if ran_junction:
            t0 = time.perf_counter()
            junction_out = junction_infer.predict(img_rgb)
            junction_ms = (time.perf_counter() - t0) * 1000.0
            module_cache['junction_lr']['last_output'] = deepcopy(junction_out)
            module_cache['junction_lr']['last_step'] = idx
            module_call_count['junction_lr'] += 1
        else:
            junction_out = _resolve_module_output_from_cache(
                'junction_lr', module_cache, module_defaults, reuse_last_outputs
            )

        if ran_straight_keep:
            t0 = time.perf_counter()
            straight_out = straight_infer.predict(img_rgb)
            straight_ms = (time.perf_counter() - t0) * 1000.0
            module_cache['straight_keep']['last_output'] = deepcopy(straight_out)
            module_cache['straight_keep']['last_step'] = idx
            module_call_count['straight_keep'] += 1
        else:
            straight_out = _resolve_module_output_from_cache(
                'straight_keep', module_cache, module_defaults, reuse_last_outputs
            )

        if ran_trigger and approach_trigger_infer is not None:
            t0 = time.perf_counter()
            trigger_out = approach_trigger_infer.predict(img_rgb)
            trigger_ms = (time.perf_counter() - t0) * 1000.0
            module_cache['approach_trigger']['last_output'] = deepcopy(trigger_out)
            module_cache['approach_trigger']['last_step'] = idx
            module_call_count['approach_trigger'] += 1
        else:
            trigger_out = _resolve_module_output_from_cache(
                'approach_trigger', module_cache, module_defaults, reuse_last_outputs
            )

        # 组装状态机输入（未运行模块使用缓存或安全默认值）
        sm_input: Dict[str, Any] = {
            'stage3': stage_out,
            'junction_lr': junction_out,
            'straight_keep': straight_out,
            'approach_trigger': trigger_out,
        }

        sm_t0 = time.perf_counter()
        sm_out = sm.update(sm_input)
        state_machine_ms = (time.perf_counter() - sm_t0) * 1000.0
        step_ms = (time.perf_counter() - step_t0) * 1000.0

        if benchmark_only:
            # warmup 帧不计入 benchmark 统计
            if idx >= benchmark_warmup:
                benchmark_preprocess_ms.append(preprocess_ms)
                if ran_stage3:
                    benchmark_stage3_ms.append(stage3_ms)
                    benchmark_call_count['stage3'] += 1
                if ran_junction:
                    benchmark_junction_ms.append(junction_ms)
                    benchmark_call_count['junction_lr'] += 1
                if ran_straight_keep:
                    benchmark_straight_ms.append(straight_ms)
                    benchmark_call_count['straight_keep'] += 1
                if ran_trigger:
                    benchmark_trigger_ms.append(trigger_ms)
                    benchmark_call_count['approach_trigger'] += 1
                benchmark_state_machine_ms.append(state_machine_ms)
                benchmark_total_step_ms.append(step_ms)
            continue

        debug = sm_out.get('debug', {}) if isinstance(sm_out.get('debug', {}), dict) else {}
        transition = debug.get('transition', None)
        transition_from = ''
        transition_to = ''
        transition_reason = ''
        if isinstance(transition, dict):
            transition_from = str(transition.get('from', ''))
            transition_to = str(transition.get('to', ''))
            transition_reason = str(transition.get('reason', ''))
            if transition_to == 'TURN':
                num_turn_entries += 1
            elif transition_to == 'RECOVER':
                num_recover_entries += 1
            if transition_from == 'TURN' and transition_to == 'RECOVER':
                reason_lc = transition_reason.lower()
                if 'turn_timeout' in reason_lc:
                    num_turn_timeout_exits += 1
                elif 'recover_signal_confirmed' in reason_lc:
                    num_recover_signal_exits += 1
                elif 'soft_exit' in reason_lc:
                    num_soft_exit_exits += 1
                elif 'recover_by_low_turn_and_low_omega' in reason_lc:
                    num_low_turn_low_omega_exits += 1
            # 新增：检测 fallback 并记录 step_idx
            if 'fallback' in transition_reason.lower():
                fallback_step_list.append(idx)

        state_now = str(sm_out.get('state', ''))
        locked_dir = sm_out.get('locked_turn_dir', None)
        locked_dir_str = '' if locked_dir is None else str(locked_dir)

        omega_raw = _safe_float(straight_out.get('omega_cmd_raw', 0.0), 0.0)
        omega_final = _safe_float(sm_out.get('omega_cmd_final', 0.0), 0.0)
        clip_applied = bool(debug.get('clip_applied', False))
        omega_before_clip = _safe_float(debug.get('omega_before_clip', omega_final), omega_final)
        omega_after_clip = _safe_float(debug.get('omega_after_clip', omega_final), omega_final)
        if clip_applied:
            num_clip_applied += 1

        state_counts[state_now] += 1
        if locked_dir_str:
            locked_turn_dir_counts[locked_dir_str] += 1

        # 新增：追踪 Turn 票数峰值
        stage_votes = debug.get('stage_votes', {})
        cur_turn_votes = int(stage_votes.get('Turn', 0)) if isinstance(stage_votes, dict) else 0
        if cur_turn_votes > turn_signal_peak_votes:
            turn_signal_peak_votes = cur_turn_votes

        # 新增：追踪 junction 首次锁定 step
        junction_candidate = debug.get('junction_candidate', None)
        if junction_candidate is not None and junction_lock_first_step is None:
            junction_lock_first_step = idx
        junction_lock_allowed = bool(debug.get('junction_lock_allowed', False))
        if first_junction_lock_allowed_step is None:
            in_approach_flow = (state_now == 'APPROACH') or (transition_from == 'APPROACH')
            if in_approach_flow and junction_lock_allowed:
                first_junction_lock_allowed_step = idx

        trigger_pred_label = str(trigger_out.get('pred_label', '')).strip() if isinstance(trigger_out, dict) else ''
        trigger_visible = bool(trigger_pred_label)
        row = {
            'step_idx': idx,
            'image_name': image_name,
            'pred_stage': stage_out.get('pred_stage', ''),
            'pred_turn_dir': junction_out.get('pred_label', ''),
            'gt_phase': gt_phase,
            'gt_turn_dir': gt_turn_dir,
            'locked_turn_dir': locked_dir_str,
            'state': state_now,
            'ran_stage3': ran_stage3,
            'ran_junction': ran_junction,
            'ran_straight_keep': ran_straight_keep,
            'ran_trigger': ran_trigger,
            'transition_from': transition_from,
            'transition_to': transition_to,
            'transition_reason': transition_reason,
            'omega_cmd_raw': omega_raw,
            'omega_cmd_final': omega_final,
            'clip_applied': clip_applied,
            'omega_before_clip': omega_before_clip,
            'omega_after_clip': omega_after_clip,
            'stage_confidence': _safe_float(stage_out.get('confidence', 0.0), 0.0),
            'junction_confidence': _safe_float(junction_out.get('confidence', 0.0), 0.0),
            'junction_candidate': '' if junction_candidate is None else str(junction_candidate),
            'junction_lock_allowed': junction_lock_allowed,
            'junction_lock_block_reason': str(debug.get('junction_lock_block_reason', '')),
            'current_recover_votes': int(_safe_float(debug.get('current_recover_votes', 0), 0)),
            'turn_exit_ready': bool(debug.get('turn_exit_ready', False)),
            'turn_soft_exit_ready': bool(debug.get('turn_soft_exit_ready', False)),
            'turn_exit_reason_final': str(debug.get('turn_exit_reason_final', '')),
            'turn_component_scale': _safe_float(debug.get('turn_component_scale', 1.0), 1.0),
            'soft_exit_triggered': bool(debug.get('soft_exit_triggered', False)),
            'straight_recover_hold_count': int(
                _safe_float(debug.get('straight_recover_hold_count', 0), 0)),
            'provisional_turn_omega_dir_hint': (
                '' if debug.get('provisional_turn_omega_dir_hint', None) is None
                else str(debug.get('provisional_turn_omega_dir_hint'))
            ),
            'provisional_turn_omega_agree': debug.get('provisional_turn_omega_agree', None),
            'provisional_turn_effective_mix_ratio': _safe_float(
                debug.get('provisional_turn_effective_mix_ratio', 0.0), 0.0),
            # 新增列
            'valid_flag': valid_flag,
            'gt_action_name': gt_action_name,
            'gt_label_name': gt_label_name,
            # 保留可选标签信息，兼容不同数据目录
            'timestamp_ns': label_row.get('timestamp_ns', ''),
            'frame_idx': label_row.get('frame_idx', ''),
            'run_name': run_name,
            # approach_trigger 可选列
            'trigger_pred': (
                trigger_pred_label if trigger_visible else ''
            ),
            'trigger_confidence': (
                _safe_float(trigger_out.get('confidence', 0.0), 0.0)
                if trigger_visible else ''
            ),
        }
        trace_rows.append(row)

        debug_row = {
            'step_idx': idx,
            'state': state_now,
            'transition': transition if isinstance(transition, dict) else None,
            'stage_votes': debug.get('stage_votes', {}),
            'junction_votes': debug.get('junction_votes', {}),
            'recover_support_count': debug.get('recover_support_count', 0),
            'omega_before_clip': omega_before_clip,
            'omega_after_clip': omega_after_clip,
            # 额外保留便于深入排查
            'clip_applied': clip_applied,
            'turn_timeout_triggered': bool(debug.get('turn_timeout_triggered', False)),
            'turn_soft_exit_ready': bool(debug.get('turn_soft_exit_ready', False)),
            'turn_exit_reason_final': str(debug.get('turn_exit_reason_final', '')),
            'turn_component_scale': _safe_float(debug.get('turn_component_scale', 1.0), 1.0),
            'soft_exit_triggered': bool(debug.get('soft_exit_triggered', False)),
            # 新增 fallback 分析字段
            'fallback_blocked_by_turn_signal': bool(
                debug.get('fallback_blocked_by_turn_signal', False)),
            'fallback_blocked_by_junction_lock': bool(
                debug.get('fallback_blocked_by_junction_lock', False)),
            'provisional_turn_omega_dir_hint': debug.get('provisional_turn_omega_dir_hint', None),
            'provisional_turn_omega_agree': debug.get('provisional_turn_omega_agree', None),
            'provisional_turn_effective_mix_ratio': _safe_float(
                debug.get('provisional_turn_effective_mix_ratio', 0.0), 0.0),
        }
        # 保留完整状态机 debug 字段，便于核查新逻辑是否实际生效
        debug_row.update(deepcopy(debug))
        # 新增：approach_trigger 信息保留到 debug_json
        if trigger_visible:
            debug_row['approach_trigger'] = deepcopy(trigger_out)
        debug_row['scheduler'] = {
            'policy': scheduler_policy,
            'state_before': state_before,
            'locked_turn_dir_before': locked_turn_before,
            'run_plan': {
                'stage3': ran_stage3,
                'junction_lr': ran_junction,
                'straight_keep': ran_straight_keep,
                'approach_trigger': ran_trigger,
            },
            'reuse_last_outputs': reuse_last_outputs,
            'cache_step': {
                'stage3': module_cache['stage3']['last_step'],
                'junction_lr': module_cache['junction_lr']['last_step'],
                'straight_keep': module_cache['straight_keep']['last_step'],
                'approach_trigger': module_cache['approach_trigger']['last_step'],
            },
        }
        debug_rows.append(debug_row)

        step_dt = max(1e-12, step_ms / 1000.0)
        recent_step_times.append(step_dt)
        if use_tqdm:
            avg_step_time = (
                sum(recent_step_times) / len(recent_step_times)
                if recent_step_times else step_dt
            )
            if avg_step_time <= 0:
                avg_step_time = step_dt
            progress.set_postfix(
                {
                    'step': idx,
                    'state': state_now if state_now else 'NA',
                    'lock': locked_dir_str if locked_dir_str else '-',
                    'ms': f'{avg_step_time * 1000.0:.1f}',
                    'fps': f'{1.0 / avg_step_time:.1f}',
                    'img': _truncate_tail(image_name, max_len=30),
                },
                refresh=False,
            )

    replay_elapsed_sec = time.perf_counter() - replay_start_time
    if use_tqdm:
        progress.close()
        avg_fps = (float(total_frames) / replay_elapsed_sec) if replay_elapsed_sec > 0 else 0.0
        vprint(
            f'[Replay] 回放循环耗时: {replay_elapsed_sec:.2f}s, '
            f'平均速度: {avg_fps:.2f} frame/s'
        )

    # 4) replay_trace.csv（保持基础功能，默认保存）
    if benchmark_only:
        warmup_used = min(benchmark_warmup, total_frames)
        measured_frames = len(benchmark_total_step_ms)
        total_stats = _summarize_timing_ms(benchmark_total_step_ms)
        stage3_stats = _summarize_timing_ms(benchmark_stage3_ms)
        junction_stats = _summarize_timing_ms(benchmark_junction_ms)
        straight_stats = _summarize_timing_ms(benchmark_straight_ms)
        trigger_stats = _summarize_timing_ms(benchmark_trigger_ms)
        sm_stats = _summarize_timing_ms(benchmark_state_machine_ms)
        io_stats = _summarize_timing_ms(benchmark_preprocess_ms)
        achieved_hz = (1000.0 / total_stats['avg']) if total_stats['avg'] > 0 else 0.0
        measured_elapsed_sec = max(1e-12, sum(benchmark_total_step_ms) / 1000.0)

        print('\n[Benchmark] 纯性能基准测试')
        print(f'  scheduler_policy: {scheduler_policy}')
        print(f'  reuse_last_outputs: {reuse_last_outputs}')
        print(f'  total frames used: {total_frames}')
        print(f'  warmup frames: {warmup_used}')
        print(f'  measured frames: {measured_frames}')

        if measured_frames <= 0:
            print('  [Warn] 测量帧为 0，请减小 --benchmark_warmup 或增大 --benchmark_steps')
            return

        print('  overall:')
        print(f'    avg_step_ms: {total_stats["avg"]:.4f}')
        print(f'    p50_step_ms: {total_stats["p50"]:.4f}')
        print(f'    p90_step_ms: {total_stats["p90"]:.4f}')
        print(f'    p95_step_ms: {total_stats["p95"]:.4f}')
        print(f'    max_step_ms: {total_stats["max"]:.4f}')
        print(f'    achieved_hz: {achieved_hz:.2f}')

        print('  module_call_count (all / measured):')
        print(f'    stage3:         {module_call_count["stage3"]} / {benchmark_call_count["stage3"]}')
        print(f'    junction_lr:    {module_call_count["junction_lr"]} / {benchmark_call_count["junction_lr"]}')
        print(f'    straight_keep:  {module_call_count["straight_keep"]} / {benchmark_call_count["straight_keep"]}')
        print(
            f'    approach_trigger: '
            f'{module_call_count["approach_trigger"]} / {benchmark_call_count["approach_trigger"]}'
        )
        print('  module_effective_hz (measured window):')
        print(f'    stage3_hz:         {_effective_hz(benchmark_call_count["stage3"], measured_elapsed_sec):.2f}')
        print(
            f'    junction_hz:       '
            f'{_effective_hz(benchmark_call_count["junction_lr"], measured_elapsed_sec):.2f}'
        )
        print(
            f'    straight_keep_hz:  '
            f'{_effective_hz(benchmark_call_count["straight_keep"], measured_elapsed_sec):.2f}'
        )
        print(
            f'    trigger_hz:        '
            f'{_effective_hz(benchmark_call_count["approach_trigger"], measured_elapsed_sec):.2f}'
        )

        print('  breakdown by module:')
        print(f'    stage3_ms:        avg={stage3_stats["avg"]:.4f}, p50={stage3_stats["p50"]:.4f}, '
              f'p90={stage3_stats["p90"]:.4f}, p95={stage3_stats["p95"]:.4f}, max={stage3_stats["max"]:.4f}')
        print(f'    junction_ms:      avg={junction_stats["avg"]:.4f}, p50={junction_stats["p50"]:.4f}, '
              f'p90={junction_stats["p90"]:.4f}, p95={junction_stats["p95"]:.4f}, max={junction_stats["max"]:.4f}')
        print(f'    straight_keep_ms: avg={straight_stats["avg"]:.4f}, p50={straight_stats["p50"]:.4f}, '
              f'p90={straight_stats["p90"]:.4f}, p95={straight_stats["p95"]:.4f}, max={straight_stats["max"]:.4f}')
        print(f'    trigger_ms:       avg={trigger_stats["avg"]:.4f}, p50={trigger_stats["p50"]:.4f}, '
              f'p90={trigger_stats["p90"]:.4f}, p95={trigger_stats["p95"]:.4f}, max={trigger_stats["max"]:.4f}')
        print(f'    state_machine_ms: avg={sm_stats["avg"]:.4f}, p50={sm_stats["p50"]:.4f}, '
              f'p90={sm_stats["p90"]:.4f}, p95={sm_stats["p95"]:.4f}, max={sm_stats["max"]:.4f}')
        print(f'    preprocess_ms:    avg={io_stats["avg"]:.4f}, p50={io_stats["p50"]:.4f}, '
              f'p90={io_stats["p90"]:.4f}, p95={io_stats["p95"]:.4f}, max={io_stats["max"]:.4f}')
        return

    # 4) replay_trace.csv（保持基础功能，默认保存）
    trace_csv = os.path.join(out_dir, 'replay_trace.csv')
    trace_fields = [
        'step_idx',
        'image_name',
        'pred_stage',
        'pred_turn_dir',
        'gt_phase',
        'gt_turn_dir',
        'locked_turn_dir',
        'state',
        'ran_stage3',
        'ran_junction',
        'ran_straight_keep',
        'ran_trigger',
        'transition_from',
        'transition_to',
        'transition_reason',
        'omega_cmd_raw',
        'omega_cmd_final',
        'clip_applied',
        'omega_before_clip',
        'omega_after_clip',
        'stage_confidence',
        'junction_confidence',
        'junction_candidate',
        'junction_lock_allowed',
        'junction_lock_block_reason',
        'current_recover_votes',
        'turn_exit_ready',
        'turn_soft_exit_ready',
        'turn_exit_reason_final',
        'turn_component_scale',
        'soft_exit_triggered',
        'straight_recover_hold_count',
        'provisional_turn_omega_dir_hint',
        'provisional_turn_omega_agree',
        'provisional_turn_effective_mix_ratio',
        # 新增列
        'valid_flag',
        'gt_action_name',
        'gt_label_name',
        # 原有可选列
        'timestamp_ns',
        'frame_idx',
        'run_name',
        # approach_trigger 可选列
        'trigger_pred',
        'trigger_confidence',
    ]
    # 出于兼容性，始终输出 replay_trace.csv；save_csv 用于显式记录与提示
    if no_save_outputs:
        trace_csv = ''
        vprint('  [i] no_save_outputs=True，跳过 replay_trace.csv 写出')
    else:
        if not save_csv:
            vprint('  [i] logging.save_csv=False，但为兼容仍输出 replay_trace.csv')
        with open(trace_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=trace_fields)
            writer.writeheader()
            for r in trace_rows:
                writer.writerow(r)

    # 5) replay_summary.json（系统级统计增强）
    state_seq = [str(r.get('state', '')) for r in trace_rows]
    unique_state_sequence = _compress_state_sequence(state_seq)
    first_turn_step = _first_step_with_state(trace_rows, 'TURN')
    first_recover_step = _first_step_with_state(trace_rows, 'RECOVER')
    turn_exit_reason_primary = _get_turn_exit_reason_primary(trace_rows)
    final_state = str(trace_rows[-1].get('state', '')) if trace_rows else ''
    turn_state_segments = _extract_state_segments(trace_rows, 'TURN')
    recover_state_segments = _extract_state_segments(trace_rows, 'RECOVER')
    turn_segment_count = len(turn_state_segments)
    recover_segment_count = len(recover_state_segments)
    straightkeep_return_count = 0
    for i in range(1, len(state_seq)):
        if state_seq[i] == 'STRAIGHTKEEP' and state_seq[i - 1] != 'STRAIGHTKEEP':
            straightkeep_return_count += 1
    if first_recover_step is not None:
        first_steps_after_first_recover = max(0, len(trace_rows) - int(first_recover_step) - 1)
    else:
        first_steps_after_first_recover = 0
    turn_duration_steps = sum(1 for s in state_seq if s == 'TURN')
    recover_duration_steps = sum(1 for s in state_seq if s == 'RECOVER')
    # 系统级成功指标：
    # - returned_to_straightkeep：最后一个去重状态是否回到 STRAIGHTKEEP
    # - task_success：方向正确 + 至少进入过 TURN/RECOVER + 最终回到 STRAIGHTKEEP
    returned_to_straightkeep = bool(unique_state_sequence) and (unique_state_sequence[-1] == 'STRAIGHTKEEP')

    # final_locked_turn_dir：仅记录 run 结束时状态，不作为主评价指标
    final_locked_turn_dir = ''
    if trace_rows:
        final_locked_turn_dir = str(trace_rows[-1].get('locked_turn_dir', ''))

    # first_locked_turn_dir：第一次出现非空 locked_turn_dir 的值
    first_locked_turn_dir = ''
    for r in trace_rows:
        d = str(r.get('locked_turn_dir', '')).strip()
        if d:
            first_locked_turn_dir = d
            break

    # most_frequent_locked_turn_dir：整个 replay 中非空 locked_turn_dir 的众数
    _locked_dir_counter: Counter = Counter()
    for r in trace_rows:
        d = str(r.get('locked_turn_dir', '')).strip()
        if d:
            _locked_dir_counter[d] += 1
    most_frequent_locked_turn_dir = ''
    if _locked_dir_counter:
        most_frequent_locked_turn_dir = _locked_dir_counter.most_common(1)[0][0]

    # 全局 gt_turn_dir：优先使用轨迹中的首个非空值；否则用 run 目录名推断
    gt_turn_dir_global = ''
    for r in trace_rows:
        d = str(r.get('gt_turn_dir', '')).strip()
        if d:
            gt_turn_dir_global = d
            break
    if not gt_turn_dir_global:
        gt_turn_dir_global = _infer_gt_turn_dir(os.path.basename(run_dir))

    # turn_dir_match 新规则：
    #   优先使用 first_locked_turn_dir 与 gt 比较
    #   若为空则退化到 most_frequent_locked_turn_dir
    #   不再依赖 final_locked_turn_dir（可能因 RECOVER->STRAIGHTKEEP 被清空）
    turn_dir_match: Optional[bool]
    if gt_turn_dir_global:
        _eval_dir = first_locked_turn_dir or most_frequent_locked_turn_dir
        if _eval_dir:
            turn_dir_match = (_eval_dir == gt_turn_dir_global)
        else:
            # 整个 replay 从未锁定过方向
            turn_dir_match = False
    else:
        turn_dir_match = None
    task_success = (
        (turn_dir_match is True)
        and (int(num_turn_entries) >= 1)
        and (int(num_recover_entries) >= 1)
        and bool(returned_to_straightkeep)
    )
    stage3_call_count = int(module_call_count['stage3'])
    junction_call_count = int(module_call_count['junction_lr'])
    straight_keep_call_count = int(module_call_count['straight_keep'])
    trigger_call_count = int(module_call_count['approach_trigger'])
    stage3_effective_hz = _effective_hz(stage3_call_count, replay_elapsed_sec)
    junction_effective_hz = _effective_hz(junction_call_count, replay_elapsed_sec)
    straight_keep_effective_hz = _effective_hz(straight_keep_call_count, replay_elapsed_sec)
    trigger_effective_hz = _effective_hz(trigger_call_count, replay_elapsed_sec)

    summary = {
        'total_steps': len(trace_rows),
        'state_counts': dict(state_counts),
        'num_turn_entries': int(num_turn_entries),
        'num_recover_entries': int(num_recover_entries),
        'locked_turn_dir_counts': dict(locked_turn_dir_counts),
        # 系统级分析字段（方向评价）
        'final_locked_turn_dir': final_locked_turn_dir,
        'first_locked_turn_dir': first_locked_turn_dir,
        'most_frequent_locked_turn_dir': most_frequent_locked_turn_dir,
        'gt_turn_dir': gt_turn_dir_global,
        'turn_dir_match': turn_dir_match,
        'first_turn_step': first_turn_step,
        'first_recover_step': first_recover_step,
        'first_steps_after_first_recover': int(first_steps_after_first_recover),
        'turn_duration_steps': int(turn_duration_steps),
        'recover_duration_steps': int(recover_duration_steps),
        'final_state': final_state,
        'turn_state_segments': turn_state_segments,
        'recover_state_segments': recover_state_segments,
        'straightkeep_return_count': int(straightkeep_return_count),
        'turn_segment_count': int(turn_segment_count),
        'recover_segment_count': int(recover_segment_count),
        'num_clip_applied': int(num_clip_applied),
        'unique_state_sequence': unique_state_sequence,
        'returned_to_straightkeep': bool(returned_to_straightkeep),
        'task_success': bool(task_success),
        'scheduler_policy': scheduler_policy,
        'reuse_last_outputs': bool(reuse_last_outputs),
        'stage3_call_count': stage3_call_count,
        'junction_call_count': junction_call_count,
        'straight_keep_call_count': straight_keep_call_count,
        'trigger_call_count': trigger_call_count,
        'stage3_effective_hz': stage3_effective_hz,
        'junction_effective_hz': junction_effective_hz,
        'straight_keep_effective_hz': straight_keep_effective_hz,
        'trigger_effective_hz': trigger_effective_hz,
        # ===== 新增：valid 过滤相关统计 =====
        'used_total_steps': len(trace_rows),
        'original_total_steps': original_total,
        'used_valid_only': valid_only,
        'used_frame_stride': frame_stride,
        'used_sample_dt_ms': sample_dt_ms,
        'skipped_invalid_steps': skipped_invalid,
        'skipped_by_sampling_steps': skipped_by_sampling,
        # ===== 新增：TURN 信号分析 =====
        'turn_signal_peak_votes': int(turn_signal_peak_votes),
        'junction_lock_first_step': junction_lock_first_step,
        'first_junction_lock_allowed_step': first_junction_lock_allowed_step,
        'fallback_step_list': fallback_step_list,
        'num_turn_timeout_exits': int(num_turn_timeout_exits),
        'num_recover_signal_exits': int(num_recover_signal_exits),
        'num_low_turn_low_omega_exits': int(num_low_turn_low_omega_exits),
        'num_soft_exit_exits': int(num_soft_exit_exits),
        'turn_exit_reason_primary': turn_exit_reason_primary,
        'thresholds': threshold_snapshot,
        # 追溯信息
        'run_dir': run_dir,
        'config_path': config_path,
        'label_fields': label_fields,
        'has_meta_json': bool(meta_json is not None),
        'meta_preview': {
            'total_frames': (meta_json or {}).get('total_frames', None),
            'valid_frames': (meta_json or {}).get('valid_frames', None),
            'duration_seconds': (meta_json or {}).get('duration_seconds', None),
        },
        'logging_flags': {
            'save_debug_json': save_debug_json,
            'save_csv': save_csv,
            'verbose': verbose,
        },
    }
    summary_json = os.path.join(out_dir, 'replay_summary.json')
    if no_save_outputs:
        summary_json = ''
        vprint('  [i] no_save_outputs=True，跳过 replay_summary.json 写出')
    else:
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # 6) replay_debug.json（可选）
    debug_json = ''
    if no_save_outputs:
        if save_debug_json:
            vprint('  [i] no_save_outputs=True，跳过 replay_debug.json 写出')
    elif save_debug_json:
        debug_json = os.path.join(out_dir, 'replay_debug.json')
        with open(debug_json, 'w', encoding='utf-8') as f:
            json.dump(debug_rows, f, ensure_ascii=False, indent=2)

    # 7) 绘制时间轴
    timeline_png = ''
    if no_save_outputs:
        vprint('  [i] no_save_outputs=True，跳过 state_timeline.png 绘制')
    else:
        timeline_png = os.path.join(out_dir, 'state_timeline.png')
        _plot_state_timeline(trace_rows, timeline_png)

    # 结束日志
    print('\n[Replay] 完成')
    if no_save_outputs:
        print('  - outputs: skipped (--no_save_outputs)')
    else:
        print(f'  - trace:    {trace_csv}')
        print(f'  - summary:  {summary_json}')
        print(f'  - timeline: {timeline_png}')
        if save_debug_json:
            print(f'  - debug:    {debug_json}')
    if fallback_step_list:
        print(f'  - fallback 发生步: {fallback_step_list}')
    if junction_lock_first_step is not None:
        print(f'  - junction 首次锁定步: {junction_lock_first_step}')
    print(f'  - scheduler: {scheduler_policy} (reuse_last_outputs={reuse_last_outputs})')
    print(
        '  - call_count: '
        f'stage3={module_call_count["stage3"]}, '
        f'junction={module_call_count["junction_lr"]}, '
        f'straight_keep={module_call_count["straight_keep"]}, '
        f'trigger={module_call_count["approach_trigger"]}'
    )
    print(f'  - Turn 票数峰值: {turn_signal_peak_votes}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='离线回放层级导航系统并生成系统级时间轴分析'
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='单个 run 目录（至少包含 images/）')
    parser.add_argument('--config', type=str,
                        default='configs/hierarchical_nav.yaml',
                        help='层级导航配置 yaml 路径')
    parser.add_argument('--out_dir', type=str, default='',
                        help='输出目录（不填则自动生成 results/replay_<run_name>）')
    parser.add_argument('--device', type=str, default=None,
                        help='推理设备，例如 cpu / cuda:0（默认自动选择）')
    parser.add_argument('--max_steps', type=int, default=0,
                        help='仅回放前 N 帧（0 表示全部）')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='步长采样：每隔 N 帧保留 1 帧（默认 1 表示不采样）')
    parser.add_argument('--sample_dt_ms', type=float, default=0.0,
                        help='时间采样间隔(ms)，>0 且存在 timestamp_ns 时优先启用')
    parser.add_argument('--scheduler_policy', type=str,
                        default='all_models',
                        choices=['all_models', 'state_conditioned_v2'],
                        help='模型调度策略：all_models(全量逐步) / state_conditioned_v2(状态依赖多速率)')
    parser.add_argument('--stage3_probe_stride', type=int, default=3,
                        help='state_conditioned_v2 下 stage3 探测步长（STRAIGHTKEEP 使用）')
    parser.add_argument('--junction_probe_stride', type=int, default=1,
                        help='state_conditioned_v2 下 junction 探测步长（APPROACH/PROVISIONAL/TURN-未锁存）')
    parser.add_argument('--straight_keep_stride', type=int, default=1,
                        help='state_conditioned_v2 下 straight_keep 步长')
    parser.add_argument('--trigger_stride', type=int, default=1,
                        help='state_conditioned_v2 下 approach_trigger 步长（STRAIGHTKEEP 使用）')
    parser.add_argument('--reuse_last_outputs', type=_parse_bool_arg, nargs='?', const=True, default=True,
                        help='模块本步未运行时是否复用最近一次输出（true/false）')
    # 新增：valid 过滤开关
    parser.add_argument('--valid_only', action='store_true', default=False,
                        help='仅回放 labels.csv 中 valid=1/true 的帧（默认回放全部）')
    parser.add_argument('--benchmark_only', action='store_true', default=False,
                        help='仅执行纯性能基准测试（不写任何回放结果文件）')
    parser.add_argument('--benchmark_warmup', type=int, default=50,
                        help='benchmark 预热帧数（不计入统计）')
    parser.add_argument('--benchmark_steps', type=int, default=500,
                        help='benchmark 总处理帧数上限')
    parser.add_argument('--no_save_outputs', action='store_true', default=False,
                        help='禁止写 trace/debug/summary/timeline 文件')
    args = parser.parse_args()
    if args.max_steps <= 0:
        args.max_steps = None
    if args.frame_stride <= 0:
        args.frame_stride = 1
    if args.sample_dt_ms <= 0:
        args.sample_dt_ms = 0.0
    if args.stage3_probe_stride <= 0:
        args.stage3_probe_stride = 1
    if args.junction_probe_stride <= 0:
        args.junction_probe_stride = 1
    if args.straight_keep_stride <= 0:
        args.straight_keep_stride = 1
    if args.trigger_stride <= 0:
        args.trigger_stride = 1
    if args.benchmark_warmup < 0:
        args.benchmark_warmup = 0
    if args.benchmark_steps <= 0:
        args.benchmark_steps = 1

    run_replay(args)


if __name__ == '__main__':
    main()
