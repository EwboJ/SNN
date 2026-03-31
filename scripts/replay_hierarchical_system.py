"""
层级导航系统离线回放脚本（系统级研究分析版，不依赖 ROS2）
=========================================================

功能：
1. 读取层级导航配置（configs/hierarchical_nav.yaml）
2. 逐帧调用三模块推理（stage3 / junction_lr / straight_keep）
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
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('缺少依赖 PyYAML，请先安装: pip install pyyaml') from exc


# 允许脚本在仓库根目录直接运行
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.corridor_module_infer import (  # noqa: E402
    JunctionLRInfer,
    Stage3Infer,
    StraightKeepInfer,
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


def _resolve_path(path_str: str, base_dir: Optional[str] = None) -> str:
    """
    解析路径：
    - 绝对路径直接返回；
    - 相对路径优先相对 base_dir，再相对仓库根目录。
    """
    if os.path.isabs(path_str):
        return path_str

    candidates: List[str] = []
    if base_dir:
        candidates.append(os.path.abspath(os.path.join(base_dir, path_str)))
    candidates.append(os.path.abspath(os.path.join(_REPO_ROOT, path_str)))
    candidates.append(os.path.abspath(path_str))

    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


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
    )


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
    state_map = {'BOOT': 0, 'STRAIGHTKEEP': 1, 'APPROACH': 2, 'TURN': 3, 'RECOVER': 4}
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
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['BOOT', 'STRAIGHTKEEP', 'APPROACH', 'TURN', 'RECOVER'])
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

    # logging 配置（真正生效）
    log_cfg = deepcopy(cfg.get('logging', {}) or {})
    save_debug_json = bool(log_cfg.get('save_debug_json', True))
    save_csv = bool(log_cfg.get('save_csv', True))
    verbose = bool(log_cfg.get('verbose', True))

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
    frames, label_fields, original_total, skipped_invalid = _collect_frames(
        run_dir, valid_only=valid_only
    )
    if args.max_steps is not None and args.max_steps > 0:
        frames = frames[:args.max_steps]

    vprint('=' * 72)
    vprint('[Replay] 层级导航系统离线回放')
    vprint(f'  配置文件:   {config_path}')
    vprint(f'  run 目录:   {run_dir}')
    vprint(f'  原始帧数:   {original_total}')
    vprint(f'  有效帧数:   {len(frames)}  (跳过无效帧: {skipped_invalid})')
    vprint(f'  valid_only: {valid_only}')
    vprint(f'  输出目录:   {out_dir}')
    vprint(f'  logging:    save_debug_json={save_debug_json}, '
           f'save_csv={save_csv}, verbose={verbose}')
    vprint('=' * 72)

    # 1) 三模块推理封装
    model_cfg = cfg.get('models', {}) or {}
    stage3_ckpt = _resolve_path(str(model_cfg.get('stage3_ckpt', '')), base_dir=cfg_dir)
    junction_ckpt = _resolve_path(str(model_cfg.get('junction_lr_ckpt', '')), base_dir=cfg_dir)
    straight_ckpt = _resolve_path(str(model_cfg.get('straight_keep_ckpt', '')), base_dir=cfg_dir)

    stage3_infer = Stage3Infer(stage3_ckpt, device=args.device)
    junction_infer = JunctionLRInfer(junction_ckpt, device=args.device)
    straight_infer = StraightKeepInfer(straight_ckpt, device=args.device)

    # 2) 层级状态机
    sm = _build_state_machine(cfg)
    sm.reset()

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

    # 新增：系统级研究分析追踪变量
    turn_signal_peak_votes = 0        # Turn 票数历史峰值
    junction_lock_first_step = None   # junction 首次锁定的 step
    fallback_step_list: List[int] = []  # 所有 fallback 发生的 step

    # 检测 labels.csv 中是否有 action_name / label_name 字段
    first_junction_lock_allowed_step: Optional[int] = None
    has_action_name = 'action_name' in label_fields
    has_label_name = 'label_name' in label_fields
    has_valid_field = 'valid' in label_fields

    for idx, fr in enumerate(frames):
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

        with Image.open(image_path) as img:
            img_rgb = img.convert('RGB')
            stage_out = stage3_infer.predict(img_rgb)
            junction_out = junction_infer.predict(img_rgb)
            straight_out = straight_infer.predict(img_rgb)

        sm_out = sm.update({
            'stage3': stage_out,
            'junction_lr': junction_out,
            'straight_keep': straight_out,
        })

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

        row = {
            'step_idx': idx,
            'image_name': image_name,
            'pred_stage': stage_out.get('pred_stage', ''),
            'pred_turn_dir': junction_out.get('pred_label', ''),
            'gt_phase': gt_phase,
            'gt_turn_dir': gt_turn_dir,
            'locked_turn_dir': locked_dir_str,
            'state': state_now,
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
            'straight_recover_hold_count': int(
                _safe_float(debug.get('straight_recover_hold_count', 0), 0)),
            # 新增列
            'valid_flag': valid_flag,
            'gt_action_name': gt_action_name,
            'gt_label_name': gt_label_name,
            # 保留可选标签信息，兼容不同数据目录
            'timestamp_ns': label_row.get('timestamp_ns', ''),
            'frame_idx': label_row.get('frame_idx', ''),
            'run_name': run_name,
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
            # 新增 fallback 分析字段
            'fallback_blocked_by_turn_signal': bool(
                debug.get('fallback_blocked_by_turn_signal', False)),
            'fallback_blocked_by_junction_lock': bool(
                debug.get('fallback_blocked_by_junction_lock', False)),
        }
        # 保留完整状态机 debug 字段，便于核查新逻辑是否实际生效
        debug_row.update(deepcopy(debug))
        debug_rows.append(debug_row)

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
        'straight_recover_hold_count',
        # 新增列
        'valid_flag',
        'gt_action_name',
        'gt_label_name',
        # 原有可选列
        'timestamp_ns',
        'frame_idx',
        'run_name',
    ]
    # 出于兼容性，始终输出 replay_trace.csv；save_csv 用于显式记录与提示
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
    turn_duration_steps = sum(1 for s in state_seq if s == 'TURN')
    recover_duration_steps = sum(1 for s in state_seq if s == 'RECOVER')

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
        'turn_duration_steps': int(turn_duration_steps),
        'recover_duration_steps': int(recover_duration_steps),
        'num_clip_applied': int(num_clip_applied),
        'unique_state_sequence': unique_state_sequence,
        # ===== 新增：valid 过滤相关统计 =====
        'used_total_steps': len(trace_rows),
        'original_total_steps': original_total,
        'used_valid_only': valid_only,
        'skipped_invalid_steps': skipped_invalid,
        # ===== 新增：TURN 信号分析 =====
        'turn_signal_peak_votes': int(turn_signal_peak_votes),
        'junction_lock_first_step': junction_lock_first_step,
        'first_junction_lock_allowed_step': first_junction_lock_allowed_step,
        'fallback_step_list': fallback_step_list,
        'num_turn_timeout_exits': int(num_turn_timeout_exits),
        'num_recover_signal_exits': int(num_recover_signal_exits),
        'num_low_turn_low_omega_exits': int(num_low_turn_low_omega_exits),
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
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 6) replay_debug.json（可选）
    debug_json = os.path.join(out_dir, 'replay_debug.json')
    if save_debug_json:
        with open(debug_json, 'w', encoding='utf-8') as f:
            json.dump(debug_rows, f, ensure_ascii=False, indent=2)
    else:
        debug_json = ''

    # 7) 绘制时间轴
    timeline_png = os.path.join(out_dir, 'state_timeline.png')
    _plot_state_timeline(trace_rows, timeline_png)

    # 结束日志
    print('\n[Replay] 完成')
    print(f'  - trace:    {trace_csv}')
    print(f'  - summary:  {summary_json}')
    print(f'  - timeline: {timeline_png}')
    if save_debug_json:
        print(f'  - debug:    {debug_json}')
    if fallback_step_list:
        print(f'  - fallback 发生步: {fallback_step_list}')
    if junction_lock_first_step is not None:
        print(f'  - junction 首次锁定步: {junction_lock_first_step}')
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
    # 新增：valid 过滤开关
    parser.add_argument('--valid_only', action='store_true', default=False,
                        help='仅回放 labels.csv 中 valid=1/true 的帧（默认回放全部）')
    args = parser.parse_args()
    if args.max_steps <= 0:
        args.max_steps = None

    run_replay(args)


if __name__ == '__main__':
    main()
