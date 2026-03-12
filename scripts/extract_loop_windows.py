"""
Full-Loop 闭环数据稀疏事件窗口提取
================================
从 full-loop 数据中提取稀疏事件窗口，
而不是直接使用整圈稠密直行帧训练。

同一条 loop 内可能有多个 turn 事件，每次 turn 独立提取窗口。

提取模式：
  A) junction_windows  — 每次 turn 的前后窗口，标签=转向方向
  B) stage_windows     — 每次 turn 的 Follow/Approach/Turn/Recover 四阶段
  C) sparse_follow     — 长直段中均匀采样若干稳定直行子片段
  D) all               — 同时提取以上全部

输入：
  data/loop_eval/<split>/<run_name>/{images/, labels.csv, meta.json}

输出：
  data/loop_sparse/<task_name>/<split>/<derived_run_name>/{images/, labels.csv, meta.json}

用法：
  # 提取全部模式
  python scripts/extract_loop_windows.py \\
      --src_root ./data/loop_eval \\
      --dst_root ./data/loop_sparse \\
      --mode all --force

  # 仅 junction_windows
  python scripts/extract_loop_windows.py \\
      --src_root ./data/loop_eval \\
      --dst_root ./data/loop_sparse \\
      --mode junction_windows
"""

import os
import sys
import csv
import json
import shutil
import random
import argparse
from collections import defaultdict


# ============================================================================
# 常量
# ============================================================================
SPLITS = ['train', 'val', 'test']
ORIG_ACTIONS = {0: 'Forward', 1: 'Backward', 2: 'Left', 3: 'Right', 4: 'Stop'}


# ============================================================================
# 数据读取
# ============================================================================

def load_labels_csv(csv_path, valid_only=True):
    """读取 labels.csv"""
    rows = []
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid = int(row.get('valid', '1'))
            if valid_only and valid == 0:
                continue
            rows.append({
                'image_name': row['image_name'],
                'action_id': int(row['action_id']),
                'action_name': row['action_name'],
                'timestamp_ns': int(row['timestamp_ns']),
                'linear_x': float(row['linear_x']),
                'angular_z': float(row['angular_z']),
                'time_diff_ms': float(row.get('time_diff_ms', '0')),
                'valid': valid,
            })
    return rows


def scan_loop_runs(src_root, valid_only=True):
    """
    扫描 src_root/<split>/<run_name>/ 下全部 loop run。

    Returns:
        list of dict: [{run_name, run_dir, split, frames}, ...]
    """
    runs = []
    for split in SPLITS:
        sp_dir = os.path.join(src_root, split)
        if not os.path.isdir(sp_dir):
            continue
        for rn in sorted(os.listdir(sp_dir)):
            rd = os.path.join(sp_dir, rn)
            if not os.path.isdir(rd):
                continue
            if not os.path.isdir(os.path.join(rd, 'images')):
                continue
            csv_path = os.path.join(rd, 'labels.csv')
            if not os.path.isfile(csv_path):
                continue
            frames = load_labels_csv(csv_path, valid_only)
            if frames:
                runs.append({
                    'run_name': rn,
                    'run_dir': rd,
                    'split': split,
                    'frames': frames,
                })
    return runs


def ns_to_ms(ns):
    return ns / 1e6


# ============================================================================
# Multi-Turn 检测 (单 loop 内多次 turn)
# ============================================================================

def detect_all_turns(frames, k=3, w_threshold=0.3, min_gap_ms=500):
    """
    检测单个 loop run 中的全部 turn 事件。

    策略:
      1. 扫描 action_name 连续 K 帧为 Left/Right 的所有段
      2. 若方法1找不够，用 angular_z 补充

    Args:
        frames: 帧列表
        k: 连续帧数
        w_threshold: angular_z 阈值
        min_gap_ms: 两次 turn 之间最小间隔 (ms)

    Returns:
        list of dict: [{turn_dir, t_turn_on_ns, t_turn_off_ns,
                        idx_on, idx_off, method, turn_id}, ...]
    """
    n = len(frames)
    if n < k:
        return []

    turns = []

    # ---- 方法1: action_name 连续段 ----
    i = 0
    while i <= n - k:
        names = [frames[j]['action_name'] for j in range(i, i + k)]
        if all(nm in ('Left', 'Right') for nm in names) and len(set(names)) == 1:
            direction = names[0].lower()
            idx_on = i
            idx_off = i + k - 1
            # 扩展到连续同方向帧
            while idx_off + 1 < n and \
                    frames[idx_off + 1]['action_name'].lower() == direction:
                idx_off += 1

            # 检查与上一个 turn 的间隔
            if turns:
                gap = ns_to_ms(frames[idx_on]['timestamp_ns'] -
                               turns[-1]['t_turn_off_ns'])
                if gap < min_gap_ms:
                    i = idx_off + 1
                    continue

            turns.append({
                'turn_dir': direction,
                't_turn_on_ns': frames[idx_on]['timestamp_ns'],
                't_turn_off_ns': frames[idx_off]['timestamp_ns'],
                'idx_on': idx_on,
                'idx_off': idx_off,
                'method': 'action_name',
            })
            i = idx_off + 1
        else:
            i += 1

    # ---- 方法2 (补充): angular_z 连续段 ----
    i = 0
    while i <= n - k:
        vals = [frames[j]['angular_z'] for j in range(i, i + k)]
        if all(abs(v) >= w_threshold for v in vals):
            positive = [v > 0 for v in vals]
            if all(positive) or not any(positive):
                direction = 'left' if vals[0] > 0 else 'right'
                idx_on = i
                idx_off = i + k - 1
                while idx_off + 1 < n:
                    nv = frames[idx_off + 1]['angular_z']
                    if abs(nv) >= w_threshold and (nv > 0) == (vals[0] > 0):
                        idx_off += 1
                    else:
                        break

                # 检查是否与已有 turn 重叠
                t_on = frames[idx_on]['timestamp_ns']
                t_off = frames[idx_off]['timestamp_ns']
                overlap = False
                for existing in turns:
                    if not (t_off < existing['t_turn_on_ns'] or
                            t_on > existing['t_turn_off_ns']):
                        overlap = True
                        break
                if not overlap:
                    if turns:
                        gap = ns_to_ms(t_on - turns[-1]['t_turn_off_ns'])
                        if gap < min_gap_ms:
                            i = idx_off + 1
                            continue
                    turns.append({
                        'turn_dir': direction,
                        't_turn_on_ns': t_on,
                        't_turn_off_ns': t_off,
                        'idx_on': idx_on,
                        'idx_off': idx_off,
                        'method': 'angular_z',
                    })
                i = idx_off + 1
            else:
                i += 1
        else:
            i += 1

    # 按时间排序并编号
    turns.sort(key=lambda t: t['t_turn_on_ns'])
    for tid, t in enumerate(turns):
        t['turn_id'] = tid

    return turns


# ============================================================================
# 直段检测
# ============================================================================

def detect_straight_segments(frames, turns, min_duration_ms=500):
    """
    检测 turn 之间的直行段。

    Returns:
        list of dict: [{idx_start, idx_end, t_start_ns, t_end_ns,
                        duration_ms}, ...]
    """
    segments = []
    turn_intervals = [(t['t_turn_on_ns'], t['t_turn_off_ns']) for t in turns]

    def in_turn(t_ns):
        for t_on, t_off in turn_intervals:
            if t_on <= t_ns <= t_off:
                return True
        return False

    seg_start = None
    for i, fr in enumerate(frames):
        is_straight = fr['action_name'] in ('Forward', 'Stop') and \
                      not in_turn(fr['timestamp_ns'])
        if is_straight:
            if seg_start is None:
                seg_start = i
        else:
            if seg_start is not None:
                seg_end = i - 1
                dur = ns_to_ms(frames[seg_end]['timestamp_ns'] -
                               frames[seg_start]['timestamp_ns'])
                if dur >= min_duration_ms:
                    segments.append({
                        'idx_start': seg_start,
                        'idx_end': seg_end,
                        't_start_ns': frames[seg_start]['timestamp_ns'],
                        't_end_ns': frames[seg_end]['timestamp_ns'],
                        'duration_ms': dur,
                    })
                seg_start = None

    if seg_start is not None:
        seg_end = len(frames) - 1
        dur = ns_to_ms(frames[seg_end]['timestamp_ns'] -
                       frames[seg_start]['timestamp_ns'])
        if dur >= min_duration_ms:
            segments.append({
                'idx_start': seg_start,
                'idx_end': seg_end,
                't_start_ns': frames[seg_start]['timestamp_ns'],
                't_end_ns': frames[seg_end]['timestamp_ns'],
                'duration_ms': dur,
            })

    return segments


# ============================================================================
# 文件操作
# ============================================================================

def copy_image(src, dst, mode='copy'):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == 'symlink':
        src_abs = os.path.abspath(src)
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src_abs, dst)
    else:
        shutil.copy2(src, dst)


def write_window_run(out_dir, frames_out, src_img_dir, copy_mode):
    """写出一个窗口 run: images/ + labels.csv + meta.json"""
    img_out = os.path.join(out_dir, 'images')
    os.makedirs(img_out, exist_ok=True)

    for fr in frames_out:
        src = os.path.join(src_img_dir, fr['image_name'])
        dst = os.path.join(img_out, fr['image_name'])
        if os.path.isfile(src):
            copy_image(src, dst, copy_mode)

    fieldnames = [
        'image_name', 'label_id', 'label_name', 'timestamp_ns',
        'linear_x', 'angular_z', 'valid',
        'orig_action_id', 'orig_action_name',
        'run_name', 'split', 'turn_id', 't_rel_ms', 'phase',
    ]
    with open(os.path.join(out_dir, 'labels.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fr in frames_out:
            w.writerow({k: fr.get(k, '') for k in fieldnames})

    label_dist = defaultdict(int)
    for fr in frames_out:
        label_dist[fr['label_name']] += 1
    meta = {
        'total_frames': len(frames_out),
        'label_distribution': dict(label_dist),
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return len(frames_out)


# ============================================================================
# 模式 A: junction_windows
# ============================================================================

def extract_junction_windows(run_info, turns, args):
    """
    提取每次 turn 的 junction window。
    标签 = 该次 turn 的转向方向。

    Returns:
        list of (derived_run_name, frames_out)
    """
    frames = run_info['frames']
    rn = run_info['run_name']
    split = run_info['split']
    results = []

    for turn in turns:
        tid = turn['turn_id']
        t_on = turn['t_turn_on_ns']
        t_off = turn['t_turn_off_ns']
        pre_ns = args.pre_turn_ms * 1e6
        post_ns = args.post_turn_ms * 1e6

        window_start = t_on - pre_ns
        window_end = t_off + post_ns
        direction = turn['turn_dir']
        lr_id = 0 if direction == 'left' else 1
        lr_name = 'Left' if direction == 'left' else 'Right'

        window_frames = []
        for fr in frames:
            t = fr['timestamp_ns']
            if t < window_start or t > window_end:
                continue
            if fr['action_name'] == 'Backward':
                continue

            t_rel = ns_to_ms(t) - ns_to_ms(t_on)
            t_ms = ns_to_ms(t)
            on_ms = ns_to_ms(t_on)
            off_ms = ns_to_ms(t_off)
            if t_ms < on_ms:
                phase = 'Pre'
            elif t_ms <= off_ms:
                phase = 'Turn'
            else:
                phase = 'Post'

            window_frames.append({
                'image_name': fr['image_name'],
                'label_id': lr_id,
                'label_name': lr_name,
                'timestamp_ns': t,
                'linear_x': fr['linear_x'],
                'angular_z': fr['angular_z'],
                'valid': fr['valid'],
                'orig_action_id': fr['action_id'],
                'orig_action_name': fr['action_name'],
                'run_name': rn,
                'split': split,
                'turn_id': tid,
                't_rel_ms': round(t_rel, 1),
                'phase': phase,
            })

        if window_frames:
            derived_name = f'{rn}_turn{tid}_{direction}'
            results.append((derived_name, window_frames))

    return results


# ============================================================================
# 模式 B: stage_windows
# ============================================================================

STAGE_LABELS = {'Follow': 0, 'Approach': 1, 'Turn': 2, 'Recover': 3}


def extract_stage_windows(run_info, turns, args):
    """
    提取每次 turn 周围的四阶段窗口。
    Follow / Approach / Turn / Recover。

    Returns:
        list of (derived_run_name, frames_out)
    """
    frames = run_info['frames']
    rn = run_info['run_name']
    split = run_info['split']
    margin_ms = 300
    results = []

    for turn in turns:
        tid = turn['turn_id']
        t_on = turn['t_turn_on_ns']
        t_off = turn['t_turn_off_ns']
        on_ms = ns_to_ms(t_on)
        off_ms = ns_to_ms(t_off)

        approach_start = on_ms - args.pre_turn_ms
        turn_start = on_ms - margin_ms
        turn_end = off_ms + margin_ms
        recover_end = off_ms + args.recover_ms
        follow_start = approach_start - 3000  # 最多向前取 3s Follow

        window_frames = []
        for fr in frames:
            t_ms = ns_to_ms(fr['timestamp_ns'])
            if t_ms < follow_start or t_ms > recover_end:
                continue
            if fr['action_name'] == 'Backward':
                continue

            if t_ms < approach_start:
                phase = 'Follow'
            elif t_ms < turn_start:
                phase = 'Approach'
            elif t_ms <= turn_end:
                phase = 'Turn'
            elif t_ms <= recover_end:
                phase = 'Recover'
            else:
                continue

            t_rel = t_ms - on_ms

            window_frames.append({
                'image_name': fr['image_name'],
                'label_id': STAGE_LABELS[phase],
                'label_name': phase,
                'timestamp_ns': fr['timestamp_ns'],
                'linear_x': fr['linear_x'],
                'angular_z': fr['angular_z'],
                'valid': fr['valid'],
                'orig_action_id': fr['action_id'],
                'orig_action_name': fr['action_name'],
                'run_name': rn,
                'split': split,
                'turn_id': tid,
                't_rel_ms': round(t_rel, 1),
                'phase': phase,
            })

        if window_frames:
            direction = turn['turn_dir']
            derived_name = f'{rn}_turn{tid}_{direction}_stage'
            results.append((derived_name, window_frames))

    return results


# ============================================================================
# 模式 C: sparse_follow
# ============================================================================

def extract_sparse_follow(run_info, turns, straight_segs, args):
    """
    对每个直行段，均匀采样 stable_follow_per_segment 个子片段，
    每个子片段长度 stable_follow_clip_ms。

    Returns:
        list of (derived_run_name, frames_out)
    """
    frames = run_info['frames']
    rn = run_info['run_name']
    split = run_info['split']
    clip_ms = args.stable_follow_clip_ms
    n_clips = args.stable_follow_per_segment
    results = []

    for seg_idx, seg in enumerate(straight_segs):
        dur = seg['duration_ms']
        if dur < clip_ms:
            continue

        t_start_ms = ns_to_ms(seg['t_start_ns'])
        available = dur - clip_ms
        if available <= 0:
            continue

        if n_clips == 1:
            offsets = [available / 2]
        else:
            offsets = [available * i / (n_clips - 1) for i in range(n_clips)]

        for clip_idx, offset in enumerate(offsets):
            clip_start_ms = t_start_ms + offset
            clip_end_ms = clip_start_ms + clip_ms

            clip_frames = []
            for fr in frames[seg['idx_start']:seg['idx_end'] + 1]:
                t_ms = ns_to_ms(fr['timestamp_ns'])
                if t_ms < clip_start_ms or t_ms > clip_end_ms:
                    continue
                if fr['action_name'] == 'Backward':
                    continue

                clip_frames.append({
                    'image_name': fr['image_name'],
                    'label_id': 0,
                    'label_name': 'Follow',
                    'timestamp_ns': fr['timestamp_ns'],
                    'linear_x': fr['linear_x'],
                    'angular_z': fr['angular_z'],
                    'valid': fr['valid'],
                    'orig_action_id': fr['action_id'],
                    'orig_action_name': fr['action_name'],
                    'run_name': rn,
                    'split': split,
                    'turn_id': -1,
                    't_rel_ms': round(t_ms - clip_start_ms, 1),
                    'phase': 'Follow',
                })

            if clip_frames:
                derived_name = f'{rn}_seg{seg_idx}_clip{clip_idx}'
                results.append((derived_name, clip_frames))

    return results


# ============================================================================
# 统一处理
# ============================================================================

MODE_NAMES = {
    'junction_windows': 'junction_windows',
    'stage_windows': 'stage_windows',
    'sparse_follow': 'sparse_follow',
}


def process_mode(mode_name, runs_by_split, all_turns, all_straights, args):
    """处理单个提取模式，保留 split 目录结构"""

    print(f'\n{"=" * 70}')
    print(f'  模式: {mode_name}')
    print(f'{"=" * 70}')

    total_derived = 0
    total_frames = 0
    label_dist = defaultdict(int)
    split_stats = {s: {'runs': 0, 'derived': 0, 'frames': 0} for s in SPLITS}
    run_details = []
    skipped = []

    for split in SPLITS:
        runs = runs_by_split.get(split, [])
        if not runs:
            continue

        for run in runs:
            rn = run['run_name']
            turns = all_turns.get(rn, [])
            straights = all_straights.get(rn, [])
            src_img_dir = os.path.join(run['run_dir'], 'images')

            if mode_name == 'junction_windows':
                derived = extract_junction_windows(run, turns, args)
            elif mode_name == 'stage_windows':
                derived = extract_stage_windows(run, turns, args)
            elif mode_name == 'sparse_follow':
                derived = extract_sparse_follow(run, turns, straights, args)
            else:
                derived = []

            if not derived:
                skipped.append({
                    'run_name': rn, 'split': split, 'reason': '无有效窗口'})
                print(f'  [跳过] [{split}] {rn}: 无有效窗口')
                continue

            for derived_name, frames_out in derived:
                # 输出到 dst_root/<task_name>/<split>/<derived_run_name>/
                out_dir = os.path.join(
                    args.dst_root, mode_name, split, derived_name)
                if args.force and os.path.exists(out_dir):
                    shutil.rmtree(out_dir)

                n_out = write_window_run(
                    out_dir, frames_out, src_img_dir, args.copy_mode)
                total_derived += 1
                total_frames += n_out
                split_stats[split]['derived'] += 1
                split_stats[split]['frames'] += n_out

                for fr in frames_out:
                    label_dist[fr['label_name']] += 1

                ldist = defaultdict(int)
                for fr in frames_out:
                    ldist[fr['label_name']] += 1
                ld_str = ', '.join(
                    f'{k}:{v}' for k, v in sorted(ldist.items()))
                print(f'  [{split:5s}] {derived_name:45s} '
                      f'{n_out:4d} 帧  ({ld_str})')

            split_stats[split]['runs'] += 1
            run_details.append({
                'run_name': rn,
                'split': split,
                'n_turns': len(turns),
                'n_straights': len(straights),
                'n_derived': len(derived),
                'total_frames': sum(len(f) for _, f in derived),
            })

    # 汇总
    print(f'\n  ── 汇总 ({mode_name}) ──')
    for sp in SPLITS:
        st = split_stats[sp]
        if st['derived'] == 0:
            continue
        print(f'  {sp:5s}: {st["runs"]:3d} src runs -> '
              f'{st["derived"]:3d} 派生  {st["frames"]:5d} 帧')
    print(f'  总计:  {total_derived} 派生  {total_frames} 帧')
    ld_str = ', '.join(f'{k}:{v}' for k, v in sorted(label_dist.items()))
    print(f'  标签分布: {ld_str}')
    if skipped:
        print(f'  跳过:     {len(skipped)} runs')

    # 保存 summary
    task_dir = os.path.join(args.dst_root, mode_name)
    os.makedirs(task_dir, exist_ok=True)
    summary = {
        'mode': mode_name,
        'total_derived_runs': total_derived,
        'total_frames': total_frames,
        'label_distribution': dict(label_dist),
        'split_stats': {k: v for k, v in split_stats.items() if v['derived']},
        'per_run': run_details,
        'skipped': skipped,
        'params': {
            'pre_turn_ms': args.pre_turn_ms,
            'post_turn_ms': args.post_turn_ms,
            'recover_ms': args.recover_ms,
            'stable_follow_clip_ms': args.stable_follow_clip_ms,
            'stable_follow_per_segment': args.stable_follow_per_segment,
            'turn_k_consecutive': args.turn_k_consecutive,
            'turn_w_threshold': args.turn_w_threshold,
        },
    }
    with open(os.path.join(task_dir, 'extract_summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'  [✓] extract_summary.json')
    print(f'{"=" * 70}')

    return summary


# ============================================================================
# 可调用入口 (供 pipeline 调用)
# ============================================================================

def run_extract_loop_windows(src_root, dst_root, mode='all',
                              valid_only=True,
                              turn_k_consecutive=3,
                              turn_w_threshold=0.3,
                              pre_turn_ms=2000,
                              post_turn_ms=1200,
                              recover_ms=1800,
                              stable_follow_clip_ms=2000,
                              stable_follow_per_segment=2,
                              copy_mode='copy',
                              seed=42,
                              force=False):
    """
    Full-loop 稀疏窗口提取核心函数。

    Returns:
        dict: 各模式的 summary
    """
    class Args:
        pass
    args = Args()
    args.src_root = src_root
    args.dst_root = dst_root
    args.mode = mode
    args.valid_only = valid_only
    args.turn_k_consecutive = turn_k_consecutive
    args.turn_w_threshold = turn_w_threshold
    args.pre_turn_ms = pre_turn_ms
    args.post_turn_ms = post_turn_ms
    args.recover_ms = recover_ms
    args.stable_follow_clip_ms = stable_follow_clip_ms
    args.stable_follow_per_segment = stable_follow_per_segment
    args.copy_mode = copy_mode
    args.seed = seed
    args.force = force

    random.seed(seed)

    print('=' * 70)
    print('  Full-Loop 闭环数据稀疏事件窗口提取')
    print('=' * 70)
    print(f'  输入:   {os.path.abspath(src_root)}')
    print(f'  输出:   {os.path.abspath(dst_root)}')
    print(f'  模式:   {mode}')
    print(f'  Turn:   K={turn_k_consecutive}, '
          f'w阈值={turn_w_threshold}')
    print(f'  窗口:   pre={pre_turn_ms}ms, '
          f'post={post_turn_ms}ms, '
          f'recover={recover_ms}ms')
    print(f'  Follow: clip={stable_follow_clip_ms}ms, '
          f'per_seg={stable_follow_per_segment}')
    print('=' * 70)

    # [1] 扫描
    print('\n[1/4] 扫描 loop runs...')
    all_runs = scan_loop_runs(src_root, valid_only)
    if not all_runs:
        print('  ✗ 未找到任何 loop run!')
        return {}

    # 按 split 分组
    runs_by_split = defaultdict(list)
    for r in all_runs:
        runs_by_split[r['split']].append(r)

    for sp in SPLITS:
        if sp in runs_by_split:
            print(f'  {sp}: {len(runs_by_split[sp])} runs')
    print(f'  总计: {len(all_runs)} 条 loop run')
    for r in all_runs:
        print(f'    [{r["split"]:5s}] {r["run_name"]}: '
              f'{len(r["frames"])} 帧')

    # [2] Turn 检测
    print('\n[2/4] 检测全部 turn 事件...')
    all_turns = {}
    total_turns = 0
    for run in all_runs:
        turns = detect_all_turns(
            run['frames'],
            k=turn_k_consecutive,
            w_threshold=turn_w_threshold)
        all_turns[run['run_name']] = turns
        total_turns += len(turns)
        if turns:
            for t in turns:
                dur = ns_to_ms(t['t_turn_off_ns'] - t['t_turn_on_ns'])
                print(f'  {run["run_name"]} turn{t["turn_id"]}: '
                      f'{t["turn_dir"]:5s} [{t["method"]:12s}] '
                      f'idx=[{t["idx_on"]:3d},{t["idx_off"]:3d}] '
                      f'dur={dur:6.0f}ms')
        else:
            print(f'  {run["run_name"]}: ✗ 未检测到 turn')
    print(f'\n  总计: {total_turns} turns / {len(all_runs)} runs')

    # [3] 直段检测
    print('\n[3/4] 检测直行段...')
    all_straights = {}
    total_segs = 0
    for run in all_runs:
        turns = all_turns.get(run['run_name'], [])
        segs = detect_straight_segments(run['frames'], turns)
        all_straights[run['run_name']] = segs
        total_segs += len(segs)
        for seg in segs:
            print(f'  {run["run_name"]} seg: '
                  f'idx=[{seg["idx_start"]:3d},{seg["idx_end"]:3d}] '
                  f'dur={seg["duration_ms"]:.0f}ms')
    print(f'\n  总计: {total_segs} 直行段')

    # [4] 提取
    print('\n[4/4] 提取事件窗口...')
    modes = list(MODE_NAMES.keys()) if mode == 'all' else [mode]
    all_summaries = {}

    for mode_name in modes:
        sm = process_mode(
            mode_name, runs_by_split, all_turns, all_straights, args)
        all_summaries[mode_name] = sm

    # 总结
    print('\n' + '=' * 70)
    print('  全部完成!')
    print('=' * 70)
    for mn, sm in all_summaries.items():
        print(f'  {mn:25s}  {sm["total_derived_runs"]:3d} 派生  '
              f'{sm["total_frames"]:5d} 帧')
    print(f'\n  输出目录: {os.path.abspath(dst_root)}/')
    print('=' * 70)

    return all_summaries


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Full-Loop 闭环数据稀疏事件窗口提取',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_root', type=str,
                        default='./data/loop_eval',
                        help='full-loop 数据根目录 (含 train/val/test/)')
    parser.add_argument('--dst_root', type=str,
                        default='./data/loop_sparse',
                        help='输出根目录')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['junction_windows', 'stage_windows',
                                 'sparse_follow', 'all'],
                        help='提取模式')
    parser.add_argument('--valid_only', type=lambda x:
                        str(x).lower() in ('true', '1', 'yes'),
                        default=True, help='丢弃 valid=0 帧')

    # Turn 检测
    parser.add_argument('--turn_k_consecutive', type=int, default=3,
                        help='连续同方向帧数阈值')
    parser.add_argument('--turn_w_threshold', type=float, default=0.3,
                        help='angular_z 绝对值阈值')

    # 时间窗口
    parser.add_argument('--pre_turn_ms', type=float, default=2000,
                        help='turn_on 前窗口 (ms)')
    parser.add_argument('--post_turn_ms', type=float, default=1200,
                        help='turn_off 后窗口 (ms)')
    parser.add_argument('--recover_ms', type=float, default=1800,
                        help='recover 窗口 (ms)')

    # Sparse follow
    parser.add_argument('--stable_follow_clip_ms', type=float, default=2000,
                        help='每个 follow 子片段长度 (ms)')
    parser.add_argument('--stable_follow_per_segment', type=int, default=2,
                        help='每个直段最多提取的子片段数')

    # 输出
    parser.add_argument('--copy_mode', type=str, default='copy',
                        choices=['copy', 'symlink'],
                        help='图片复制方式')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true',
                        help='覆盖已有输出')

    args = parser.parse_args()

    run_extract_loop_windows(
        src_root=args.src_root,
        dst_root=args.dst_root,
        mode=args.mode,
        valid_only=args.valid_only,
        turn_k_consecutive=args.turn_k_consecutive,
        turn_w_threshold=args.turn_w_threshold,
        pre_turn_ms=args.pre_turn_ms,
        post_turn_ms=args.post_turn_ms,
        recover_ms=args.recover_ms,
        stable_follow_clip_ms=args.stable_follow_clip_ms,
        stable_follow_per_segment=args.stable_follow_per_segment,
        copy_mode=args.copy_mode,
        seed=args.seed,
        force=args.force,
    )


if __name__ == '__main__':
    main()
