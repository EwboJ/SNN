"""
统一任务数据集核查可视化脚本
========================================
支持核查：
  - junction_lr_v1      (二分类: Left/Right)
  - stage3_v1           (三阶段: Approach/Turn/Recover)
  - stage4_v1           (四阶段: Follow/Approach/Turn/Recover)
  - action3_balanced_v1 (三分类: Left/Straight/Right)
  - straight_keep_reg_v1 (回归: angular_z, Correcting/Settled)
  - loop_sparse/*       (junction_windows / stage_windows / sparse_follow)

对每个任务数据集：
  1. 随机抽若干 run
  2. 拼接连续帧 strip 预览图
  3. 标注 label_name / phase / t_rel_ms / angular_z
  4. angular_z 时间轴图 (phase 背景色)
  5. 输出 preview png + verify_summary.json

用法：
  # 自动检测任务类型 (junction_lr)
  python scripts/verify_task_datasets.py \\
      --data_root ./data/stage1/junction_lr_v1

  # 核查 stage3 三阶段
  python scripts/verify_task_datasets.py \\
      --data_root ./data/stage1/stage3_v1

  # 核查 stage4 四阶段
  python scripts/verify_task_datasets.py \\
      --data_root ./data/stage1/stage4_v1

  # 核查 straight_keep 回归
  python scripts/verify_task_datasets.py \\
      --data_root ./data/straight_keep/straight_keep_reg_v1

  # 核查 loop_sparse (模式目录)
  python scripts/verify_task_datasets.py \\
      --data_root ./data/loop_sparse/junction_windows

  # 指定 split 和数量
  python scripts/verify_task_datasets.py \\
      --data_root ./data/stage1/stage3_v1 \\
      --split train --max_runs 8 --frames_per_run 30
"""

import os
import sys
import csv
import json
import math
import random
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# 中文字体
for font in ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']:
    try:
        plt.rcParams['font.sans-serif'] = [font] + \
            plt.rcParams['font.sans-serif']
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 120})


# ============================================================================
# 颜色方案
# ============================================================================

PHASE_COLORS = {
    'Follow':      '#42A5F5',
    'Approach':    '#FFA726',
    'Turn':        '#EF5350',
    'Recover':     '#66BB6A',
    'Pre':         '#AB47BC',
    'Post':        '#78909C',
    'Correcting':  '#FF7043',
    'Settled':     '#26A69A',
}

LABEL_COLORS = {
    'Left':        '#EF5350',
    'Right':       '#42A5F5',
    'Straight':    '#66BB6A',
    'Forward':     '#66BB6A',
    'Stop':        '#BDBDBD',
    'Follow':      '#42A5F5',
    'Approach':    '#FFA726',
    'Turn':        '#EF5350',
    'Recover':     '#66BB6A',
}


# ============================================================================
# 任务类型检测
# ============================================================================

KNOWN_TASKS = {
    'junction_lr':        {'type': 'classification', 'has_phase': True},
    'stage3':             {'type': 'classification', 'has_phase': True},
    'stage4':             {'type': 'classification', 'has_phase': True},
    'action3':            {'type': 'classification', 'has_phase': True},
    'straight_keep_reg':  {'type': 'regression',     'has_phase': True},
    'junction_windows':   {'type': 'classification', 'has_phase': True},
    'stage_windows':      {'type': 'classification', 'has_phase': True},
    'sparse_follow':      {'type': 'classification', 'has_phase': False},
}


def detect_task_type(data_root):
    """从目录名推断任务类型"""
    base = os.path.basename(data_root.rstrip('/\\'))
    for key in KNOWN_TASKS:
        if key in base:
            return key
    # 尝试父目录 (loop_sparse/junction_windows)
    parent = os.path.basename(os.path.dirname(data_root.rstrip('/\\')))
    if 'loop_sparse' in parent:
        for key in KNOWN_TASKS:
            if key in base:
                return key
    return 'unknown'


# ============================================================================
# 数据读取
# ============================================================================

def load_run_csv(run_dir):
    """读取 run 的 labels.csv，返回 dict list"""
    csv_path = os.path.join(run_dir, 'labels.csv')
    if not os.path.isfile(csv_path):
        return []
    frames = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            frames.append(row)
    return frames


def scan_runs(data_root, split=None):
    """扫描 data_root/<split>/<run>/ 目录"""
    runs = []
    splits = [split] if split else ['train', 'val', 'test']
    for sp in splits:
        sp_dir = os.path.join(data_root, sp)
        if not os.path.isdir(sp_dir):
            continue
        for rn in sorted(os.listdir(sp_dir)):
            rd = os.path.join(sp_dir, rn)
            if not os.path.isdir(rd):
                continue
            frames = load_run_csv(rd)
            if frames:
                runs.append({
                    'run_name': rn,
                    'run_dir': rd,
                    'split': sp,
                    'frames': frames,
                })
    return runs


# ============================================================================
# Strip 图 (帧缩略图拼接)
# ============================================================================

def make_strip_image(run_info, max_frames=30, thumb_size=56):
    """将连续帧拼成一行 strip 图"""
    frames = run_info['frames']
    img_dir = os.path.join(run_info['run_dir'], 'images')

    n = len(frames)
    if n > max_frames:
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
    else:
        indices = list(range(n))

    thumbnails = []
    infos = []
    for idx in indices:
        fr = frames[idx]
        img_path = os.path.join(img_dir, fr['image_name'])
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
            thumbnails.append(np.array(img))
        except Exception:
            thumbnails.append(
                np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8))
        infos.append(fr)

    if not thumbnails:
        return None, []

    strip = np.concatenate(thumbnails, axis=1)
    return strip, infos


# ============================================================================
# Preview 图
# ============================================================================

def plot_preview(run_info, task_type, out_path, frames_per_run=25):
    """生成 preview 图: strip + 标签条 + phase 条 (可选)"""
    rn = run_info['run_name']
    sp = run_info['split']
    frames = run_info['frames']
    task_info = KNOWN_TASKS.get(task_type, {})
    has_phase = task_info.get('has_phase', False)
    is_regression = task_info.get('type') == 'regression'

    strip, infos = make_strip_image(
        run_info, max_frames=frames_per_run, thumb_size=56)
    if strip is None:
        return

    n_shown = len(infos)

    # 图布局
    n_rows = 2  # strip + label bar
    ratios = [3, 0.6]
    if has_phase:
        n_rows += 1
        ratios.append(0.4)
    if is_regression:
        n_rows += 1
        ratios.append(1.2)

    fig_w = max(12, n_shown * 0.55)
    fig_h = sum(ratios) * 0.9 + 1.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(n_rows, 1, height_ratios=ratios)
    row_idx = 0

    # ---- Strip 图 ----
    ax_img = fig.add_subplot(gs[row_idx])
    row_idx += 1
    ax_img.imshow(strip, aspect='auto')
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # 标题
    title_parts = [f'{rn} [{sp}]']
    if task_type == 'junction_lr':
        lr = infos[0].get('label_name', '?')
        title_parts.append(f'方向: {lr}')
    elif task_type == 'straight_keep_reg':
        title_parts.append('回归 angular_z')
    title_parts.append(f'({len(frames)} 帧, 显示 {n_shown})')
    ax_img.set_title('  '.join(title_parts),
                      fontsize=11, fontweight='bold')

    # 标签边框色
    thumb_w = strip.shape[1] / n_shown
    for i, info in enumerate(infos):
        label_name = info.get('label_name', '')
        color = LABEL_COLORS.get(label_name, '#BDBDBD')
        x = i * thumb_w
        rect = plt.Rectangle(
            (x, 0), thumb_w - 1, strip.shape[0] - 1,
            linewidth=2, edgecolor=color, facecolor='none')
        ax_img.add_patch(rect)

    # ---- 标签条 ----
    ax_lbl = fig.add_subplot(gs[row_idx])
    row_idx += 1
    ax_lbl.set_xlim(0, n_shown)
    ax_lbl.set_ylim(0, 1)
    ax_lbl.set_yticks([])

    for i, info in enumerate(infos):
        label_name = info.get('label_name', '')
        color = LABEL_COLORS.get(label_name, '#BDBDBD')
        ax_lbl.barh(0.5, 1, left=i, height=0.8, color=color, alpha=0.8)
        # 标注 t_rel_ms
        t_rel = info.get('t_rel_ms', '')
        if t_rel:
            try:
                t_val = float(t_rel)
                if i % max(1, n_shown // 10) == 0:
                    ax_lbl.text(i + 0.5, -0.15, f'{t_val:.0f}',
                                ha='center', va='top', fontsize=6,
                                color='#555')
            except (ValueError, TypeError):
                pass

    ax_lbl.set_xlabel('t_rel_ms', fontsize=8)

    # 标签图例
    seen_labels = set()
    lbl_handles = []
    for info in infos:
        ln = info.get('label_name', '')
        if ln and ln not in seen_labels:
            seen_labels.add(ln)
            lbl_handles.append(mpatches.Patch(
                color=LABEL_COLORS.get(ln, '#BDBDBD'), label=ln))
    if lbl_handles:
        ax_lbl.legend(handles=lbl_handles, loc='upper right', fontsize=7,
                      ncol=len(lbl_handles), framealpha=0.8)

    # ---- Phase 条 ----
    if has_phase:
        ax_ph = fig.add_subplot(gs[row_idx])
        row_idx += 1
        ax_ph.set_xlim(0, n_shown)
        ax_ph.set_ylim(0, 1)
        ax_ph.set_yticks([])

        for i, info in enumerate(infos):
            phase = info.get('phase', '')
            color = PHASE_COLORS.get(phase, '#EEEEEE')
            ax_ph.barh(0.5, 1, left=i, height=0.8, color=color, alpha=0.85)

        seen_ph = set()
        ph_handles = []
        for info in infos:
            ph = info.get('phase', '')
            if ph and ph not in seen_ph:
                seen_ph.add(ph)
                ph_handles.append(mpatches.Patch(
                    color=PHASE_COLORS.get(ph, '#EEE'), label=ph))
        if ph_handles:
            ax_ph.legend(handles=ph_handles, loc='upper right', fontsize=7,
                         ncol=len(ph_handles), framealpha=0.8)
        ax_ph.set_xlabel('Phase', fontsize=8)

    # ---- 回归: angular_z 小折线 ----
    if is_regression:
        ax_az = fig.add_subplot(gs[row_idx])
        row_idx += 1
        az_vals = []
        t_vals = []
        for info in infos:
            try:
                az_vals.append(float(info.get('angular_z', 0)))
                t_vals.append(float(info.get('t_rel_ms', 0)))
            except (ValueError, TypeError):
                az_vals.append(0)
                t_vals.append(0)

        ax_az.plot(range(n_shown), az_vals, color='#E65100',
                   linewidth=1.2, marker='.', markersize=3)
        ax_az.axhline(y=0, color='#999', linewidth=0.5, linestyle='--')
        ax_az.set_xlim(0, n_shown - 1)
        ax_az.set_ylabel('ω', fontsize=9)
        ax_az.set_xlabel('帧索引', fontsize=8)
        ax_az.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Timeline 图 (详细时间轴)
# ============================================================================

def plot_timeline(run_info, task_type, out_path):
    """绘制 angular_z 时间轴，背景色 = phase"""
    frames = run_info['frames']
    rn = run_info['run_name']
    task_info = KNOWN_TASKS.get(task_type, {})
    is_regression = task_info.get('type') == 'regression'

    t_rels = []
    ang_zs = []
    phases = []
    labels = []
    for fr in frames:
        try:
            t_rels.append(float(fr.get('t_rel_ms', 0)))
            ang_zs.append(float(fr.get('angular_z', 0)))
        except (ValueError, TypeError):
            t_rels.append(0)
            ang_zs.append(0)
        phases.append(fr.get('phase', ''))
        labels.append(fr.get('label_name', ''))

    if not t_rels:
        return

    fig, ax = plt.subplots(figsize=(12, 3.5))

    # 背景色块按 phase
    if any(phases):
        prev_phase = phases[0]
        seg_start = t_rels[0]
        for i in range(1, len(phases)):
            if phases[i] != prev_phase or i == len(phases) - 1:
                seg_end = t_rels[i]
                color = PHASE_COLORS.get(prev_phase, '#F5F5F5')
                ax.axvspan(seg_start, seg_end, alpha=0.2, color=color)
                prev_phase = phases[i]
                seg_start = seg_end

    # angular_z 曲线
    ax.plot(t_rels, ang_zs, color='#333', linewidth=1, alpha=0.7,
            label='angular_z')
    scatter_colors = [LABEL_COLORS.get(l, '#999') for l in labels]
    ax.scatter(t_rels, ang_zs, c=scatter_colors,
               s=12, zorder=3, alpha=0.8)

    # t=0 标记 (仅对 turn-based 任务)
    if task_type in ('junction_lr', 'stage3', 'stage4',
                     'junction_windows', 'stage_windows', 'action3'):
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.6,
                   label='t_turn_on')

    # 回归: 标注均值线
    if is_regression and ang_zs:
        az_mean = sum(ang_zs) / len(ang_zs)
        ax.axhline(y=az_mean, color='#1565C0', linestyle=':',
                   alpha=0.6, label=f'ω̄={az_mean:.3f}')

    ax.set_xlabel('t_rel_ms', fontsize=10)
    ax.set_ylabel('angular_z', fontsize=10)

    title = f'{rn} — angular_z 时间轴'
    if is_regression:
        title += f'  (n={len(frames)})'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图例
    ph_handles = []
    for ph, color in PHASE_COLORS.items():
        if ph in phases:
            ph_handles.append(
                mpatches.Patch(color=color, alpha=0.4, label=ph))
    if task_type in ('junction_lr', 'stage4', 'junction_windows',
                     'stage_windows', 'action3'):
        ph_handles.append(plt.Line2D(
            [0], [0], color='red', linestyle='--', label='t_turn_on'))
    if is_regression and ang_zs:
        ph_handles.append(plt.Line2D(
            [0], [0], color='#1565C0', linestyle=':', label=f'ω̄'))
    if ph_handles:
        ax.legend(handles=ph_handles, fontsize=7, loc='upper left', ncol=3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 统计检查
# ============================================================================

def compute_run_stats(run_info, task_type):
    """计算单个 run 的统计信息"""
    frames = run_info['frames']
    label_dist = defaultdict(int)
    phase_dist = defaultdict(int)
    az_vals = []
    t_rels = []

    for fr in frames:
        label_dist[fr.get('label_name', '')] += 1
        phase_dist[fr.get('phase', '')] += 1
        try:
            az_vals.append(float(fr.get('angular_z', 0)))
        except (ValueError, TypeError):
            pass
        try:
            t_rels.append(float(fr.get('t_rel_ms', 0)))
        except (ValueError, TypeError):
            pass

    stats = {
        'run_name': run_info['run_name'],
        'split': run_info['split'],
        'total_frames': len(frames),
        'label_distribution': dict(label_dist),
        'phase_distribution': dict(phase_dist),
    }

    if t_rels:
        stats['t_rel_range_ms'] = [
            round(min(t_rels), 1), round(max(t_rels), 1)]
        stats['duration_ms'] = round(max(t_rels) - min(t_rels), 1)

    if az_vals:
        az_mean = sum(az_vals) / len(az_vals)
        az_std = math.sqrt(
            sum((v - az_mean) ** 2 for v in az_vals) / len(az_vals))
        az_abs_mean = sum(abs(v) for v in az_vals) / len(az_vals)
        stats['angular_z_stats'] = {
            'mean': round(az_mean, 4),
            'std': round(az_std, 4),
            'abs_mean': round(az_abs_mean, 4),
            'min': round(min(az_vals), 4),
            'max': round(max(az_vals), 4),
        }

    # 异常检测
    warnings = []
    if len(frames) < 3:
        warnings.append('帧数过少 (<3)')
    if task_type == 'junction_lr':
        if len(label_dist) > 1:
            warnings.append(f'单 run 存在多个标签: {list(label_dist.keys())}')
    if task_type == 'straight_keep_reg' and az_vals:
        if max(abs(v) for v in az_vals) > 1.5:
            warnings.append(f'angular_z 最大值异常 (>{1.5})')
    if t_rels and (max(t_rels) - min(t_rels)) < 100:
        warnings.append('时间跨度过短 (<100ms)')

    stats['warnings'] = warnings
    return stats


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='统一任务数据集核查可视化 '
                    '(junction_lr / stage3 / stage4 / '
                    'straight_keep_reg / loop_sparse)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str, required=True,
                        help='任务数据集根目录 (含 train/val/test/)')
    parser.add_argument('--task', type=str, default=None,
                        choices=list(KNOWN_TASKS.keys()) + [None],
                        help='强制指定任务类型 (默认自动检测)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='输出目录 (默认 results/verify_<task>)')
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'val', 'test'],
                        help='仅检查指定 split (默认全部)')
    parser.add_argument('--max_runs', type=int, default=6,
                        help='最多随机抽取几个 run')
    parser.add_argument('--frames_per_run', type=int, default=25,
                        help='strip 图最多显示帧数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_timeline', action='store_true',
                        help='跳过 timeline 图生成 (加速)')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅列出 run, 不生成图')

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # 检测任务类型
    task_type = args.task or detect_task_type(args.data_root)
    task_info = KNOWN_TASKS.get(task_type, {})

    if args.out_dir is None:
        args.out_dir = f'./results/verify_{task_type}'

    type_label = {
        'junction_lr':        '路口二分类 (Left/Right)',
        'stage3':             '三阶段分类 (Approach/Turn/Recover)',
        'stage4':             '四阶段分类 (Follow/Approach/Turn/Recover)',
        'action3':            '三分类 (Forward/Left/Right)',
        'straight_keep_reg':  '直行纠偏回归 (angular_z)',
        'junction_windows':   'Loop 路口窗口',
        'stage_windows':      'Loop 阶段窗口',
        'sparse_follow':      'Loop 稀疏直行',
        'unknown':            '未知类型',
    }.get(task_type, task_type)

    print('=' * 70)
    print('  统一任务数据集核查')
    print('=' * 70)
    print(f'  数据:     {os.path.abspath(args.data_root)}')
    print(f'  任务:     {task_type} ({type_label})')
    print(f'  类型:     {task_info.get("type", "unknown")}')
    print(f'  输出:     {os.path.abspath(args.out_dir)}')
    print(f'  Split:    {args.split or "全部"}')
    print(f'  Max runs: {args.max_runs}')
    print('=' * 70)

    # [1] 扫描
    print('\n[1/4] 扫描 runs...')
    all_runs = scan_runs(args.data_root, split=args.split)
    if not all_runs:
        print('  ✗ 未找到任何 run!')
        sys.exit(1)

    split_counts = defaultdict(int)
    total_frames = 0
    for r in all_runs:
        split_counts[r['split']] += 1
        total_frames += len(r['frames'])
    for sp, cnt in sorted(split_counts.items()):
        print(f'  {sp}: {cnt} runs')
    print(f'  总计: {len(all_runs)} runs, {total_frames} 帧')

    # [2] 全局统计
    print('\n[2/4] 全局统计...')
    global_label_dist = defaultdict(int)
    global_phase_dist = defaultdict(int)
    global_az = []
    for r in all_runs:
        for fr in r['frames']:
            global_label_dist[fr.get('label_name', '')] += 1
            ph = fr.get('phase', '')
            if ph:
                global_phase_dist[ph] += 1
            try:
                global_az.append(float(fr.get('angular_z', 0)))
            except (ValueError, TypeError):
                pass

    print(f'  标签分布:')
    for k, v in sorted(global_label_dist.items(),
                       key=lambda x: -x[1]):
        pct = 100 * v / max(total_frames, 1)
        print(f'    {k:15s}: {v:6d} ({pct:5.1f}%)')

    if global_phase_dist:
        print(f'  Phase 分布:')
        for k, v in sorted(global_phase_dist.items(),
                           key=lambda x: -x[1]):
            pct = 100 * v / max(total_frames, 1)
            print(f'    {k:15s}: {v:6d} ({pct:5.1f}%)')

    if global_az:
        az_mean = sum(global_az) / len(global_az)
        az_abs_mean = sum(abs(v) for v in global_az) / len(global_az)
        print(f'  angular_z:')
        print(f'    mean={az_mean:.4f}  |mean|={az_abs_mean:.4f}  '
              f'min={min(global_az):.4f}  max={max(global_az):.4f}')

    # [3] 随机采样
    if len(all_runs) > args.max_runs:
        selected = rng.sample(all_runs, args.max_runs)
    else:
        selected = list(all_runs)

    print(f'\n[3/4] 选中 {len(selected)} 个 run:')
    for r in selected:
        print(f'  {r["split"]:5s}  {r["run_name"]:40s}  {len(r["frames"])} 帧')

    if args.dry_run:
        print('\n[DRY RUN] 不生成图, 退出。')
        return

    # [4] 生成可视化
    print(f'\n[4/4] 生成可视化...')
    os.makedirs(args.out_dir, exist_ok=True)

    # 按 split 汇总 label_distribution 和帧数
    per_split_stats = {}
    for sp in sorted(split_counts.keys()):
        sp_label_dist = defaultdict(int)
        sp_frames = 0
        sp_runs = 0
        for r in all_runs:
            if r['split'] != sp:
                continue
            sp_runs += 1
            sp_frames += len(r['frames'])
            for fr in r['frames']:
                sp_label_dist[fr.get('label_name', '')] += 1
        per_split_stats[sp] = {
            'runs': sp_runs,
            'frames': sp_frames,
            'label_distribution': dict(sp_label_dist),
        }

    verify_summary = {
        'task': task_type,
        'task_type_info': task_info,
        'data_root': os.path.abspath(args.data_root),
        'global_stats': {
            'total_runs': len(all_runs),
            'total_frames': total_frames,
            'split_counts': dict(split_counts),
            'label_distribution': dict(global_label_dist),
            'phase_distribution': dict(global_phase_dist),
        },
        'per_split': per_split_stats,
        'runs_checked': [],
    }

    for run in selected:
        rn = run['run_name']
        print(f'  处理 {rn}...')

        # Preview strip
        preview_path = os.path.join(args.out_dir, f'preview_{rn}.png')
        plot_preview(run, task_type, preview_path,
                     frames_per_run=args.frames_per_run)
        print(f'    [✓] preview_{rn}.png')

        # Timeline
        if not args.skip_timeline:
            timeline_path = os.path.join(
                args.out_dir, f'timeline_{rn}.png')
            plot_timeline(run, task_type, timeline_path)
            print(f'    [✓] timeline_{rn}.png')

        # 统计
        stats = compute_run_stats(run, task_type)
        verify_summary['runs_checked'].append(stats)

        if stats['warnings']:
            for w in stats['warnings']:
                print(f'    ⚠ {w}')

    # 保存 summary
    summary_path = os.path.join(args.out_dir, 'verify_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(verify_summary, f, indent=2, ensure_ascii=False)
    print(f'\n  [✓] verify_summary.json')

    # 列出全部输出
    print(f'\n{"=" * 70}')
    print(f'  核查完成! 共 {len(selected)} 个 run')
    print(f'{"=" * 70}')
    for fn in sorted(os.listdir(args.out_dir)):
        fp = os.path.join(args.out_dir, fn)
        sz = os.path.getsize(fp) / 1024
        print(f'  {fn:50s}  {sz:6.1f} KB')

    # 核查要点提示
    print(f'\n  🔍 人工核查提示 [{type_label}]:')
    if task_type == 'junction_lr':
        print(f'    - preview: 每个 run 的标签应该一致 (全 Left 或全 Right)')
        print(f'    - timeline: angular_z 在 t=0 附近应有对应方向的变化')
    elif task_type == 'stage3':
        print(f'    - phase 条: Approach→Turn→Recover 三段是否连贯 (应无 Follow)')
        print(f'    - timeline: angular_z 在 Turn 阶段应有明显变化')
        print(f'    - label 条: 三类颜色分布是否与 phase 一致')
    elif task_type == 'stage4':
        print(f'    - phase 条: Follow→Approach→Turn→Recover 是否连贯')
        print(f'    - timeline: angular_z 在 Turn 阶段应有明显变化')
    elif task_type == 'straight_keep_reg':
        print(f'    - phase 条: Correcting 后接 Settled (末尾角速度趋零)')
        print(f'    - timeline: angular_z 应逐渐趋近 0')
        print(f'    - 折线图: 检查纠偏过程是否合理')
    elif task_type in ('junction_windows', 'stage_windows'):
        print(f'    - strip 图: 帧内容与标签是否对应')
        print(f'    - timeline: turn 窗口对齐是否正确')
    elif task_type == 'sparse_follow':
        print(f'    - strip 图: 应为稳定直行画面, 无明显转弯')
        print(f'    - timeline: angular_z 应普遍接近 0')

    # 全局警告汇总
    all_warnings = []
    for rc in verify_summary['runs_checked']:
        for w in rc.get('warnings', []):
            all_warnings.append(f'{rc["run_name"]}: {w}')
    if all_warnings:
        print(f'\n  ⚠ 全局警告 ({len(all_warnings)} 条):')
        for aw in all_warnings[:20]:
            print(f'    {aw}')
        if len(all_warnings) > 20:
            print(f'    ... 及 {len(all_warnings) - 20} 条更多')

    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
