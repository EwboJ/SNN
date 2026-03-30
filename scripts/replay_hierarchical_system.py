"""
层级导航系统离线回放脚本（不依赖 ROS2）
=========================================

功能：
1) 读取层级导航配置（configs/hierarchical_nav.yaml）
2) 逐帧调用三个推理模块（stage3 / junction_lr / straight_keep）
3) 调用层级状态机 update() 得到最终控制输出
4) 生成回放时间轴与统计结果

输入 run 目录要求：
  - 必须包含 images/
  - 可选 labels.csv（若存在，优先按其 image_name 顺序回放）
  - 可选 metadata（如 meta.json）

输出文件：
  - replay_trace.csv
  - replay_summary.json
  - state_timeline.png

示例命令：
  # 在单个 run 上调试
  python scripts/replay_hierarchical_system.py ^
      --run_dir data/corridor/test/J1_left_r02 ^
      --config configs/hierarchical_nav.yaml ^
      --out_dir results/replay_J1_left_r02 ^
      --device cuda:0

  # 先小样本快速检查
  python scripts/replay_hierarchical_system.py ^
      --run_dir data/corridor/test/J1_left_r02 ^
      --max_steps 120
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


# 让脚本可直接从仓库根目录运行
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
    """用于文件名自然排序（0002.jpg 在 0010.jpg 前）。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', text)]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _resolve_path(path_str: str, base_dir: Optional[str] = None) -> str:
    """
    解析路径：
    - 绝对路径直接返回
    - 相对路径优先相对 base_dir，再相对仓库根目录
    """
    if os.path.isabs(path_str):
        return path_str

    candidates = []
    if base_dir:
        candidates.append(os.path.abspath(os.path.join(base_dir, path_str)))
    candidates.append(os.path.abspath(os.path.join(_REPO_ROOT, path_str)))
    candidates.append(os.path.abspath(path_str))

    for p in candidates:
        if os.path.exists(p):
            return p
    # 即使不存在，也返回最符合工程习惯的路径
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


def _collect_frames(run_dir: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    收集回放帧：
    1) 若 labels.csv 存在且有 image_name，优先按 labels 顺序；
    2) 否则按 images/ 文件名自然排序。
    """
    images_dir = os.path.join(run_dir, 'images')
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f'run 目录缺少 images/: {run_dir}')

    labels_csv = os.path.join(run_dir, 'labels.csv')
    label_rows, label_fields = _load_labels_rows(labels_csv)

    frames: List[Dict[str, Any]] = []
    if label_rows and ('image_name' in label_rows[0]):
        for row in label_rows:
            image_name = str(row.get('image_name', '')).strip()
            if not image_name:
                continue
            image_path = os.path.join(images_dir, image_name)
            if not os.path.isfile(image_path):
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
            frames.append({
                'image_name': n,
                'image_path': os.path.join(images_dir, n),
                'label_row': {},
            })

    if not frames:
        raise RuntimeError(f'没有可用图像帧: {run_dir}')
    return frames, label_fields


def _build_state_machine(cfg: Dict[str, Any]) -> HierarchicalNavigatorStateMachine:
    sm_cfg = deepcopy(cfg.get('state_machine', {}) or {})
    turn_cfg = deepcopy(cfg.get('turn_control', {}) or {})
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
        left_turn_omega=turn_cfg.get('left_omega', 1.2),
        right_turn_omega=turn_cfg.get('right_omega', -1.2),
    )


def _plot_state_timeline(trace_rows: List[Dict[str, Any]], out_png: str) -> None:
    """
    绘制系统时间轴图：
    - 状态随时间变化
    - stage3 预测
    - junction_lr 预测（同时显示锁存方向）
    - straight_keep 原始角速度与最终角速度
    """
    if not trace_rows:
        # 空数据时输出一张占位图，避免流程中断
        Image.new('RGB', (1280, 720), color=(255, 255, 255)).save(out_png)
        return

    # 延迟导入，避免环境中 matplotlib/numpy 二进制冲突导致脚本无法启动
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        # 兜底：仍然写出 png，提示需要 matplotlib 才能获得完整可视化
        from PIL import ImageDraw
        canvas = Image.new('RGB', (1400, 900), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        text = [
            'state_timeline.png (fallback)',
            '当前环境无法导入 matplotlib，已输出占位图。',
            f'错误: {e}',
            '',
            '建议安装兼容版本后重跑，以生成完整时间轴曲线图。',
            '例如：pip install --upgrade matplotlib numpy',
        ]
        y = 40
        for line in text:
            draw.text((40, y), line, fill=(20, 20, 20))
            y += 36
        canvas.save(out_png)
        return

    steps = [int(r['step_idx']) for r in trace_rows]

    state_map = {
        'BOOT': 0,
        'STRAIGHTKEEP': 1,
        'APPROACH': 2,
        'TURN': 3,
        'RECOVER': 4,
    }
    stage_map = {
        'Approach': 0,
        'Turn': 1,
        'Recover': 2,
    }
    turn_map = {
        'Left': 0,
        'Right': 1,
    }

    state_vals = [state_map.get(str(r.get('state', '')), -1) for r in trace_rows]
    stage_vals = [stage_map.get(str(r.get('pred_stage', '')), -1) for r in trace_rows]
    pred_turn_vals = [turn_map.get(str(r.get('pred_turn_dir', '')), -1) for r in trace_rows]
    locked_turn_vals = [turn_map.get(str(r.get('locked_turn_dir', '')), -1) for r in trace_rows]
    omega_raw_vals = [_safe_float(r.get('omega_cmd_raw', 0.0), 0.0) for r in trace_rows]
    omega_final_vals = [_safe_float(r.get('omega_cmd_final', 0.0), 0.0) for r in trace_rows]

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    ax = axes[0]
    ax.plot(steps, state_vals, drawstyle='steps-post', lw=1.8, color='#1E88E5')
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['BOOT', 'STRAIGHTKEEP', 'APPROACH', 'TURN', 'RECOVER'])
    ax.set_ylabel('State')
    ax.set_title('Hierarchical State Timeline')
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(steps, stage_vals, drawstyle='steps-post', lw=1.6, color='#43A047')
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

    run_dir = _resolve_path(args.run_dir, base_dir=_REPO_ROOT)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f'run_dir 不存在: {run_dir}')

    # 输出目录默认按 run 名命名
    if args.out_dir:
        out_dir = _resolve_path(args.out_dir, base_dir=_REPO_ROOT)
    else:
        out_dir = os.path.join(_REPO_ROOT, 'results',
                               f'replay_{os.path.basename(run_dir)}')
    os.makedirs(out_dir, exist_ok=True)

    # 读取可选 metadata
    meta_json = _load_optional_json(os.path.join(run_dir, 'meta.json'))

    # 收集帧
    frames, label_fields = _collect_frames(run_dir)
    if args.max_steps is not None and args.max_steps > 0:
        frames = frames[:args.max_steps]

    print('=' * 72)
    print('[Replay] 层级导航系统离线回放')
    print(f'  配置文件:   {config_path}')
    print(f'  run 目录:   {run_dir}')
    print(f'  帧数量:     {len(frames)}')
    print(f'  输出目录:   {out_dir}')
    print('=' * 72)

    # ===== 1) 加载三个推理模块 =====
    model_cfg = cfg.get('models', {}) or {}
    stage3_ckpt = _resolve_path(str(model_cfg.get('stage3_ckpt', '')), base_dir=cfg_dir)
    junction_ckpt = _resolve_path(str(model_cfg.get('junction_lr_ckpt', '')), base_dir=cfg_dir)
    straight_ckpt = _resolve_path(str(model_cfg.get('straight_keep_ckpt', '')), base_dir=cfg_dir)

    stage3_infer = Stage3Infer(stage3_ckpt, device=args.device)
    junction_infer = JunctionLRInfer(junction_ckpt, device=args.device)
    straight_infer = StraightKeepInfer(straight_ckpt, device=args.device)

    # ===== 2) 加载状态机 =====
    sm = _build_state_machine(cfg)
    sm.reset()

    # ===== 3) 逐帧回放 =====
    trace_rows: List[Dict[str, Any]] = []
    state_counts = Counter()
    locked_turn_dir_counts = Counter()
    num_turn_entries = 0
    num_recover_entries = 0

    for idx, fr in enumerate(frames):
        image_name = fr['image_name']
        image_path = fr['image_path']
        label_row = fr.get('label_row', {}) or {}

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

        state_now = str(sm_out.get('state', ''))
        locked_dir = sm_out.get('locked_turn_dir', None)
        locked_dir_str = '' if locked_dir is None else str(locked_dir)
        omega_raw = _safe_float(straight_out.get('omega_cmd_raw', 0.0), 0.0)
        omega_final = _safe_float(sm_out.get('omega_cmd_final', 0.0), 0.0)

        transition = (sm_out.get('debug', {}) or {}).get('transition', None)
        if isinstance(transition, dict):
            to_state = str(transition.get('to', ''))
            if to_state == 'TURN':
                num_turn_entries += 1
            elif to_state == 'RECOVER':
                num_recover_entries += 1

        state_counts[state_now] += 1
        if locked_dir_str:
            locked_turn_dir_counts[locked_dir_str] += 1

        row = {
            'step_idx': idx,
            'image_name': image_name,
            'pred_stage': stage_out.get('pred_stage', ''),
            'pred_turn_dir': junction_out.get('pred_label', ''),
            'locked_turn_dir': locked_dir_str,
            'state': state_now,
            'omega_cmd_raw': omega_raw,
            'omega_cmd_final': omega_final,
            'stage_confidence': _safe_float(stage_out.get('confidence', 0.0), 0.0),
            'junction_confidence': _safe_float(junction_out.get('confidence', 0.0), 0.0),
            # 扩展字段：尽量保留 labels.csv 信息，便于后续分析
            'timestamp_ns': label_row.get('timestamp_ns', ''),
            'frame_idx': label_row.get('frame_idx', ''),
            'run_name': label_row.get('run_name', os.path.basename(run_dir)),
        }
        trace_rows.append(row)

    # ===== 4) 保存 replay_trace.csv =====
    trace_csv = os.path.join(out_dir, 'replay_trace.csv')
    trace_fields = [
        'step_idx',
        'image_name',
        'pred_stage',
        'pred_turn_dir',
        'locked_turn_dir',
        'state',
        'omega_cmd_raw',
        'omega_cmd_final',
        'stage_confidence',
        'junction_confidence',
        'timestamp_ns',
        'frame_idx',
        'run_name',
    ]
    with open(trace_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=trace_fields)
        writer.writeheader()
        for r in trace_rows:
            writer.writerow(r)

    # ===== 5) 保存 replay_summary.json =====
    summary = {
        'total_steps': len(trace_rows),
        'state_counts': dict(state_counts),
        'num_turn_entries': int(num_turn_entries),
        'num_recover_entries': int(num_recover_entries),
        'locked_turn_dir_counts': dict(locked_turn_dir_counts),
        # 额外上下文信息（便于追溯）
        'run_dir': run_dir,
        'config_path': config_path,
        'label_fields': label_fields,
        'has_meta_json': bool(meta_json is not None),
        'meta_preview': {
            'total_frames': (meta_json or {}).get('total_frames', None),
            'valid_frames': (meta_json or {}).get('valid_frames', None),
            'duration_seconds': (meta_json or {}).get('duration_seconds', None),
        },
    }
    summary_json = os.path.join(out_dir, 'replay_summary.json')
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ===== 6) 绘制 state_timeline.png =====
    timeline_png = os.path.join(out_dir, 'state_timeline.png')
    _plot_state_timeline(trace_rows, timeline_png)

    print('\n[Replay] 完成')
    print(f'  - trace:    {trace_csv}')
    print(f'  - summary:  {summary_json}')
    print(f'  - timeline: {timeline_png}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='离线回放层级导航系统并生成时间轴分析'
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
    args = parser.parse_args()
    if args.max_steps <= 0:
        args.max_steps = None

    run_replay(args)


if __name__ == '__main__':
    main()
