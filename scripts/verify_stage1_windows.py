"""
阶段一派生数据核查可视化脚本
================================
核查 derive_stage1_datasets.py 生成的派生结果，
通过 strip 图 + 时间轴彩条直观检查窗口是否合理。

用法:
  # 核查 junction_lr
  python scripts/verify_stage1_windows.py --data_root ./data/stage1/junction_lr_v1 --out_dir ./results/verify_junction_lr

  # 核查 stage4
  python scripts/verify_stage1_windows.py --data_root ./data/stage1/stage4_v1 --out_dir ./results/verify_stage4

  # 核查 action3
  python scripts/verify_stage1_windows.py --data_root ./data/stage1/action3_balanced_v1 --out_dir ./results/verify_action3

  # 指定 split 和数量
  python scripts/verify_stage1_windows.py --data_root ./data/stage1/stage4_v1 --split train --max_runs 4 --frames_per_run 20
"""

import os
import sys
import csv
import json
import random
import argparse
from collections import defaultdict, OrderedDict

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
        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 120})


# ============================================================================
# 颜色方案
# ============================================================================
PHASE_COLORS = {
    'Follow':   '#42A5F5',  # 蓝
    'Approach': '#FFA726',  # 橙
    'Turn':     '#EF5350',  # 红
    'Recover':  '#66BB6A',  # 绿
    'Pre':      '#AB47BC',  # 紫
    'Post':     '#78909C',  # 灰蓝
}

LABEL_COLORS = {
    'Left':     '#EF5350',
    'Right':    '#42A5F5',
    'Straight': '#66BB6A',
    'Forward':  '#66BB6A',
    'Stop':     '#BDBDBD',
    'Follow':   '#42A5F5',
    'Approach': '#FFA726',
    'Turn':     '#EF5350',
    'Recover':  '#66BB6A',
}


# ============================================================================
# 数据读取
# ============================================================================

def load_derived_run(run_dir):
    """读取派生 run 的 labels.csv"""
    csv_path = os.path.join(run_dir, 'labels.csv')
    if not os.path.isfile(csv_path):
        return []
    frames = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            frames.append(row)
    return frames


def scan_derived_runs(data_root, split=None):
    """扫描派生数据中的 run"""
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
            frames = load_derived_run(rd)
            if frames:
                runs.append({
                    'run_name': rn,
                    'run_dir': rd,
                    'split': sp,
                    'frames': frames,
                })
    return runs


def detect_task_type(data_root):
    """从目录名推断任务类型"""
    base = os.path.basename(data_root.rstrip('/\\'))
    if 'junction_lr' in base:
        return 'junction_lr'
    elif 'stage4' in base:
        return 'stage4'
    elif 'action3' in base:
        return 'action3'
    return 'unknown'


# ============================================================================
# Strip 图 (帧拼接带)
# ============================================================================

def make_strip_image(run_info, max_frames=30, thumb_size=64):
    """
    将连续帧拼成一行 strip 图。

    Returns:
        (strip_img: np.array, frame_infos: list of dict)
    """
    frames = run_info['frames']
    img_dir = os.path.join(run_info['run_dir'], 'images')

    # 等间距采样
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
            thumbnails.append(np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8))
        infos.append(fr)

    if not thumbnails:
        return None, []

    strip = np.concatenate(thumbnails, axis=1)
    return strip, infos


# ============================================================================
# Preview 图
# ============================================================================

def plot_preview(run_info, task_type, out_path, frames_per_run=25):
    """
    生成 preview 图: strip + 标注 + 时间轴彩条。
    """
    rn = run_info['run_name']
    sp = run_info['split']
    frames = run_info['frames']

    strip, infos = make_strip_image(run_info, max_frames=frames_per_run,
                                     thumb_size=56)
    if strip is None:
        return

    n_shown = len(infos)

    # 图高度
    has_timeline = task_type in ('stage4', 'junction_lr', 'action3')
    fig_h = 4.5 if has_timeline else 3.0
    fig_w = max(12, n_shown * 0.55)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(3 if has_timeline else 2, 1,
                           height_ratios=[3, 0.6, 0.4] if has_timeline
                           else [3, 0.6])

    # ---- Strip 图 ----
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(strip, aspect='auto')
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # 标题
    if task_type == 'junction_lr':
        lr = infos[0].get('label_name', '?')
        ax_img.set_title(
            f'{rn} [{sp}]  方向: {lr}  ({len(frames)} 帧, 显示 {n_shown})',
            fontsize=12, fontweight='bold')
    else:
        ax_img.set_title(
            f'{rn} [{sp}]  ({len(frames)} 帧, 显示 {n_shown})',
            fontsize=12, fontweight='bold')

    # 标签边框色
    thumb_w = strip.shape[1] / n_shown
    for i, info in enumerate(infos):
        label_name = info.get('label_name', '')
        color = LABEL_COLORS.get(label_name, '#BDBDBD')
        x = i * thumb_w
        rect = plt.Rectangle((x, 0), thumb_w - 1, strip.shape[0] - 1,
                              linewidth=2, edgecolor=color, facecolor='none')
        ax_img.add_patch(rect)

    # ---- 标签条 ----
    ax_lbl = fig.add_subplot(gs[1])
    ax_lbl.set_xlim(0, n_shown)
    ax_lbl.set_ylim(0, 1)
    ax_lbl.set_yticks([])

    for i, info in enumerate(infos):
        label_name = info.get('label_name', '')
        orig = info.get('orig_action_name', '')
        color = LABEL_COLORS.get(label_name, '#BDBDBD')
        ax_lbl.barh(0.5, 1, left=i, height=0.8, color=color, alpha=0.8)
        # 标注 t_rel_ms
        t_rel = info.get('t_rel_ms', '')
        if t_rel:
            try:
                t_val = float(t_rel)
                if i % max(1, n_shown // 12) == 0:
                    ax_lbl.text(i + 0.5, -0.15, f'{t_val:.0f}',
                                ha='center', va='top', fontsize=6,
                                color='#555')
            except (ValueError, TypeError):
                pass

    ax_lbl.set_xlabel('t_rel_ms', fontsize=8)

    # 图例
    seen = set()
    handles = []
    for info in infos:
        ln = info.get('label_name', '')
        if ln and ln not in seen:
            seen.add(ln)
            handles.append(mpatches.Patch(
                color=LABEL_COLORS.get(ln, '#BDBDBD'), label=ln))
    ax_lbl.legend(handles=handles, loc='upper right', fontsize=7,
                  ncol=len(handles), framealpha=0.8)

    # ---- 时间轴 (phase 彩条) ----
    if has_timeline:
        ax_tl = fig.add_subplot(gs[2])
        ax_tl.set_xlim(0, n_shown)
        ax_tl.set_ylim(0, 1)
        ax_tl.set_yticks([])

        for i, info in enumerate(infos):
            phase = info.get('phase', '')
            color = PHASE_COLORS.get(phase, '#EEEEEE')
            ax_tl.barh(0.5, 1, left=i, height=0.8, color=color, alpha=0.85)

        phase_seen = set()
        ph_handles = []
        for info in infos:
            ph = info.get('phase', '')
            if ph and ph not in phase_seen:
                phase_seen.add(ph)
                ph_handles.append(mpatches.Patch(
                    color=PHASE_COLORS.get(ph, '#EEE'), label=ph))
        ax_tl.legend(handles=ph_handles, loc='upper right', fontsize=7,
                     ncol=len(ph_handles), framealpha=0.8)
        ax_tl.set_xlabel('Phase', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# Timeline 图 (详细时间轴)
# ============================================================================

def plot_timeline(run_info, task_type, out_path):
    """
    绘制详细时间轴:
    x = t_rel_ms, y = 原始 angular_z, 背景色 = phase 区段。
    """
    frames = run_info['frames']
    rn = run_info['run_name']

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
    if phases:
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
    ax.scatter(t_rels, ang_zs, c=[LABEL_COLORS.get(l, '#999') for l in labels],
               s=12, zorder=3, alpha=0.8)

    # t=0 标记
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.6,
               label='t_turn_on')

    ax.set_xlabel('t_rel_ms', fontsize=10)
    ax.set_ylabel('angular_z', fontsize=10)
    ax.set_title(f'{rn} — angular_z 时间轴', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 图例
    ph_handles = []
    for ph, color in PHASE_COLORS.items():
        if ph in phases:
            ph_handles.append(mpatches.Patch(color=color, alpha=0.4, label=ph))
    ph_handles.append(plt.Line2D([0], [0], color='red', linestyle='--',
                                  label='t_turn_on'))
    ax.legend(handles=ph_handles, fontsize=7, loc='upper left', ncol=3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='阶段一派生数据核查可视化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str, required=True,
                        help='派生任务根目录 (如 data/stage1/junction_lr_v1)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='输出目录 (默认 results/verify_<task>)')
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'val', 'test'],
                        help='仅检查指定 split (默认全部)')
    parser.add_argument('--max_runs', type=int, default=6,
                        help='最多检查几个 run')
    parser.add_argument('--frames_per_run', type=int, default=25,
                        help='每个 run 的 strip 图最多显示帧数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true',
                        help='仅列出将要检查的 run, 不生成图')

    args = parser.parse_args()
    rng = random.Random(args.seed)

    task_type = detect_task_type(args.data_root)
    if args.out_dir is None:
        args.out_dir = f'./results/verify_{task_type}'

    print('=' * 70)
    print(f'  阶段一派生数据核查')
    print('=' * 70)
    print(f'  数据: {os.path.abspath(args.data_root)}')
    print(f'  任务: {task_type}')
    print(f'  输出: {os.path.abspath(args.out_dir)}')
    print(f'  Split: {args.split or "全部"}')
    print(f'  Max runs: {args.max_runs}')
    print('=' * 70)

    # 扫描
    print('\n[1/3] 扫描派生 runs...')
    all_runs = scan_derived_runs(args.data_root, split=args.split)
    if not all_runs:
        print('  ✗ 未找到任何 run!')
        sys.exit(1)

    split_counts = defaultdict(int)
    for r in all_runs:
        split_counts[r['split']] += 1
    for sp, cnt in sorted(split_counts.items()):
        print(f'  {sp}: {cnt} runs')
    print(f'  总计: {len(all_runs)} runs')

    # 随机采样
    if len(all_runs) > args.max_runs:
        selected = rng.sample(all_runs, args.max_runs)
    else:
        selected = list(all_runs)

    print(f'\n[2/3] 选中 {len(selected)} 个 run:')
    for r in selected:
        print(f'  {r["split"]:5s}  {r["run_name"]:20s}  {len(r["frames"])} 帧')

    if args.dry_run:
        print('\n[DRY RUN] 不生成图, 退出。')
        return

    # 生成图
    print(f'\n[3/3] 生成可视化...')
    os.makedirs(args.out_dir, exist_ok=True)

    verify_summary = {
        'task': task_type,
        'data_root': args.data_root,
        'runs_checked': [],
    }

    for run in selected:
        rn = run['run_name']
        print(f'  处理 {rn}...')

        # Preview strip
        preview_path = os.path.join(args.out_dir, f'preview_{rn}.png')
        plot_preview(run, task_type, preview_path,
                     frames_per_run=args.frames_per_run)
        print(f'    [✓] {preview_path}')

        # Timeline
        timeline_path = os.path.join(args.out_dir, f'timeline_{rn}.png')
        plot_timeline(run, task_type, timeline_path)
        print(f'    [✓] {timeline_path}')

        # 统计
        label_dist = defaultdict(int)
        phase_dist = defaultdict(int)
        for fr in run['frames']:
            label_dist[fr.get('label_name', '')] += 1
            phase_dist[fr.get('phase', '')] += 1

        t_rels = []
        for fr in run['frames']:
            try:
                t_rels.append(float(fr.get('t_rel_ms', 0)))
            except (ValueError, TypeError):
                pass

        run_info = {
            'run_name': rn,
            'split': run['split'],
            'total_frames': len(run['frames']),
            'label_distribution': dict(label_dist),
            'phase_distribution': dict(phase_dist),
            't_rel_range_ms': [round(min(t_rels), 1), round(max(t_rels), 1)]
            if t_rels else [],
        }
        verify_summary['runs_checked'].append(run_info)

    # 保存 summary
    summary_path = os.path.join(args.out_dir, 'verify_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(verify_summary, f, indent=2, ensure_ascii=False)
    print(f'\n  [✓] {summary_path}')

    # 列出全部输出
    print(f'\n{"=" * 70}')
    print(f'  核查完成! 共 {len(selected)} 个 run')
    print(f'{"=" * 70}')
    for fn in sorted(os.listdir(args.out_dir)):
        fp = os.path.join(args.out_dir, fn)
        sz = os.path.getsize(fp) / 1024
        print(f'  {fn:45s}  {sz:6.1f} KB')
    print(f'{"=" * 70}')
    print(f'  请人工检查 preview_*.png 和 timeline_*.png:')
    print(f'    - strip 图: 帧内容是否与标签一致')
    print(f'    - 标签条: 颜色分布是否合理')
    print(f'    - 时间轴: angular_z 在 t=0 前后是否有明显转弯')
    print(f'    - phase 彩条: Follow→Approach→Turn→Recover 是否连贯')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
