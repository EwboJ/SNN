"""
走廊导航阶段一数据集派生脚本
================================
从已完成 train/val/test 划分的原始 corridor run 数据中,
派生三个任务数据集:

  A) action3_balanced  — 三分类 (Left/Straight/Right), Straight 分层采样
  B) junction_lr       — 二分类 (Left/Right), 仅保留转弯窗口
  C) stage4            — 四分类 (Follow/Approach/Turn/Recover), 时序阶段

用法:
  # 派生全部任务
  python scripts/derive_stage1_datasets.py --src_root ./data/corridor --dst_root ./data/stage1 --task all

  # 仅派生 action3_balanced
  python scripts/derive_stage1_datasets.py --src_root ./data/corridor --dst_root ./data/stage1 --task action3_balanced

  # 自定义参数
  python scripts/derive_stage1_datasets.py --src_root ./data/corridor --dst_root ./data/stage1 --task all --turn_k_consecutive 3 --pre_turn_ms 1200 --straight_ratio_cap 1.5
"""

import os
import sys
import re
import csv
import json
import shutil
import random
import argparse
from collections import defaultdict, OrderedDict
from pathlib import Path


# ============================================================================
# 常量
# ============================================================================
SPLITS = ['train', 'val', 'test']

# 原始 action 定义
ORIG_ACTIONS = {0: 'Forward', 1: 'Backward', 2: 'Left', 3: 'Right', 4: 'Stop'}

# 目录名解析: J1_left_r03
_RE_RUN = re.compile(r'^J(?P<jid>\d+)_(?P<turn>left|right)_r(?P<rep>\d+)$', re.I)


# ============================================================================
# 数据读取
# ============================================================================
def load_run_labels(run_dir, valid_only=True):
    """
    从 run_dir/labels.csv 读取所有帧。

    Returns:
        list of dict, 每个元素含: image_name, action_id, action_name,
            timestamp_ns, linear_x, angular_z, time_diff_ms, valid, idx
    """
    csv_path = os.path.join(run_dir, 'labels.csv')
    if not os.path.isfile(csv_path):
        return []

    frames = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            valid = int(row.get('valid', '1'))
            if valid_only and valid == 0:
                continue
            frames.append({
                'image_name': row['image_name'],
                'action_id': int(row['action_id']),
                'action_name': row['action_name'],
                'timestamp_ns': int(row['timestamp_ns']),
                'linear_x': float(row['linear_x']),
                'angular_z': float(row['angular_z']),
                'time_diff_ms': float(row['time_diff_ms']),
                'valid': valid,
                'idx': i,
            })
    return frames


def scan_all_runs(src_root, valid_only=True):
    """
    扫描 src_root 下 train/val/test 的所有 run。

    Returns:
        list of dict: [{run_name, run_dir, split, frames, junction_id,
                         turn_dir, rep_id}, ...]
    """
    all_runs = []
    for split in SPLITS:
        sp_dir = os.path.join(src_root, split)
        if not os.path.isdir(sp_dir):
            continue
        for rn in sorted(os.listdir(sp_dir)):
            rd = os.path.join(sp_dir, rn)
            if not os.path.isdir(rd):
                continue
            img_dir = os.path.join(rd, 'images')
            if not os.path.isdir(img_dir):
                continue

            frames = load_run_labels(rd, valid_only)
            if not frames:
                continue

            # 解析目录名
            m = _RE_RUN.match(rn)
            jid = int(m.group('jid')) if m else -1
            tdir = m.group('turn').lower() if m else 'unknown'
            rep = int(m.group('rep')) if m else -1

            all_runs.append({
                'run_name': rn,
                'run_dir': rd,
                'split': split,
                'frames': frames,
                'junction_id': jid,
                'turn_dir': tdir,
                'rep_id': rep,
            })
    return all_runs


# ============================================================================
# Turn 事件检测
# ============================================================================
def detect_turn_event(frames, k=3, w_threshold=0.3):
    """
    检测单个 run 的 turn 事件。

    策略:
      1. 首选: action_name 连续 K 帧为 Left 或 Right
      2. 备选: |angular_z| >= threshold 且符号一致连续 K 帧

    Returns:
        dict: {turn_dir, t_turn_on_ns, t_turn_off_ns,
               idx_on, idx_off, method} 或 None
    """
    n = len(frames)

    # ---- 方法1: action_name 连续 K 帧 ----
    def find_consecutive_action(target_actions):
        for i in range(n - k + 1):
            names = [frames[j]['action_name'] for j in range(i, i + k)]
            if all(nm in target_actions for nm in names):
                # 找到起始; 扩展到 turn 结束
                direction = names[0]
                end = i + k - 1
                while end + 1 < n and frames[end + 1]['action_name'] == direction:
                    end += 1
                return i, end, direction
        return None

    result = find_consecutive_action({'Left', 'Right'})
    if result:
        idx_on, idx_off, direction = result
        return {
            'turn_dir': direction.lower(),
            't_turn_on_ns': frames[idx_on]['timestamp_ns'],
            't_turn_off_ns': frames[idx_off]['timestamp_ns'],
            'idx_on': idx_on,
            'idx_off': idx_off,
            'method': 'action_name',
        }

    # ---- 方法2: angular_z 连续 K 帧 ----
    for i in range(n - k + 1):
        vals = [frames[j]['angular_z'] for j in range(i, i + k)]
        if all(abs(v) >= w_threshold for v in vals):
            signs = [v > 0 for v in vals]
            if all(signs) or not any(signs):
                direction = 'left' if vals[0] > 0 else 'right'
                end = i + k - 1
                while end + 1 < n:
                    nv = frames[end + 1]['angular_z']
                    if abs(nv) >= w_threshold and (nv > 0) == (vals[0] > 0):
                        end += 1
                    else:
                        break
                return {
                    'turn_dir': direction,
                    't_turn_on_ns': frames[i]['timestamp_ns'],
                    't_turn_off_ns': frames[end]['timestamp_ns'],
                    'idx_on': i,
                    'idx_off': end,
                    'method': 'angular_z',
                }

    return None


def ns_to_ms(ns):
    return ns / 1e6


# ============================================================================
# 帧标注 helper
# ============================================================================
def classify_frame_phase(t_ns, t_on_ns, t_off_ns, pre_ms, post_ms, recover_ms):
    """将帧按时间分类为阶段"""
    t_ms = ns_to_ms(t_ns)
    on_ms = ns_to_ms(t_on_ns)
    off_ms = ns_to_ms(t_off_ns)

    if t_ms < on_ms - pre_ms:
        return 'Follow'
    elif t_ms < on_ms:
        return 'Approach'
    elif t_ms <= off_ms:
        return 'Turn'
    elif t_ms <= off_ms + recover_ms:
        return 'Recover'
    else:
        return 'Post'


def copy_image(src_path, dst_path, mode='symlink'):
    """复制或链接单张图片"""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if mode == 'symlink':
        src_abs = os.path.abspath(src_path)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.symlink(src_abs, dst_path)
    else:
        shutil.copy2(src_path, dst_path)


def write_derived_run(out_dir, frames_out, src_img_dir, copy_mode):
    """
    写出一个派生 run 的 images/ + labels.csv + meta.json。

    Args:
        out_dir: 输出目录
        frames_out: list of dict (包含输出标签的帧)
        src_img_dir: 源图片目录
        copy_mode: 'copy' or 'symlink'
    """
    img_out = os.path.join(out_dir, 'images')
    os.makedirs(img_out, exist_ok=True)

    # 复制图片
    for fr in frames_out:
        src = os.path.join(src_img_dir, fr['image_name'])
        dst = os.path.join(img_out, fr['image_name'])
        if os.path.isfile(src):
            copy_image(src, dst, copy_mode)

    # labels.csv
    fieldnames = [
        'image_name', 'label_id', 'label_name', 'timestamp_ns',
        'linear_x', 'angular_z', 'valid',
        'orig_action_id', 'orig_action_name',
        'run_name', 'split', 't_rel_ms', 'phase'
    ]
    csv_path = os.path.join(out_dir, 'labels.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fr in frames_out:
            row = {k: fr.get(k, '') for k in fieldnames}
            w.writerow(row)

    # meta.json
    label_dist = defaultdict(int)
    for fr in frames_out:
        label_dist[fr['label_name']] += 1

    meta = {
        'total_frames': len(frames_out),
        'valid_frames': len(frames_out),
        'label_distribution': dict(label_dist),
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return len(frames_out)


# ============================================================================
# 任务 A: action3_balanced
# ============================================================================
ACTION3_MAP = {'Left': (0, 'Left'), 'Right': (2, 'Right'),
               'Forward': (1, 'Straight'), 'Stop': (1, 'Straight')}


def derive_action3_balanced(run_info, turn_event, args):
    """
    派生三分类数据集。
    保留所有 Left/Right, Straight 分层采样。

    Returns:
        list of derived frame dicts, or None if skipped
    """
    frames = run_info['frames']
    if turn_event is None:
        return None

    t_on = turn_event['t_turn_on_ns']
    t_off = turn_event['t_turn_off_ns']
    pre_ms = args.pre_turn_ms
    recover_ms = args.recover_ms
    max_follow = args.max_follow_frames
    ratio_cap = args.straight_ratio_cap
    rng = random.Random(args.seed)

    left_right_frames = []
    straight_follow = []
    straight_approach = []
    straight_recover = []

    for fr in frames:
        act = fr['action_name']
        if act == 'Backward':
            continue

        if act not in ACTION3_MAP:
            continue

        label_id, label_name = ACTION3_MAP[act]
        phase = classify_frame_phase(
            fr['timestamp_ns'], t_on, t_off, pre_ms, 0, recover_ms)
        t_rel = ns_to_ms(fr['timestamp_ns']) - ns_to_ms(t_on)

        out = {
            'image_name': fr['image_name'],
            'label_id': label_id,
            'label_name': label_name,
            'timestamp_ns': fr['timestamp_ns'],
            'linear_x': fr['linear_x'],
            'angular_z': fr['angular_z'],
            'valid': fr['valid'],
            'orig_action_id': fr['action_id'],
            'orig_action_name': fr['action_name'],
            'run_name': run_info['run_name'],
            'split': run_info['split'],
            't_rel_ms': round(t_rel, 1),
            'phase': phase,
        }

        if act in ('Left', 'Right'):
            left_right_frames.append(out)
        else:
            # Straight: 按阶段分桶
            if phase == 'Follow':
                straight_follow.append(out)
            elif phase == 'Approach':
                straight_approach.append(out)
            elif phase == 'Recover':
                straight_recover.append(out)
            # Turn 阶段的 Forward/Stop 通常很少,也保留
            else:
                straight_approach.append(out)  # 归入 approach 桶

    # Follow 限制
    if len(straight_follow) > max_follow:
        rng.shuffle(straight_follow)
        straight_follow = straight_follow[:max_follow]

    # 合并 Straight
    all_straight = straight_follow + straight_approach + straight_recover

    # 比例限制: Straight <= ratio_cap * (Left + Right)
    n_lr = len(left_right_frames)
    max_straight = int(ratio_cap * max(n_lr, 1))
    if len(all_straight) > max_straight:
        # 优先保留 Approach 和 Recover, 删 Follow
        approach_recover = [f for f in all_straight if f['phase'] != 'Follow']
        follow_only = [f for f in all_straight if f['phase'] == 'Follow']
        budget = max_straight - len(approach_recover)
        if budget > 0 and follow_only:
            rng.shuffle(follow_only)
            follow_only = follow_only[:budget]
        elif budget <= 0:
            follow_only = []
            rng.shuffle(approach_recover)
            approach_recover = approach_recover[:max_straight]
        all_straight = approach_recover + follow_only

    result = left_right_frames + all_straight
    result.sort(key=lambda x: x['timestamp_ns'])
    return result


# ============================================================================
# 任务 B: junction_lr
# ============================================================================
def derive_junction_lr(run_info, turn_event, args):
    """
    派生二分类数据集 (Left/Right)。
    仅保留 [t_on - pre_ms, t_off + post_ms] 窗口内的帧。
    所有帧标签统一继承该 run 的转向方向。

    Returns:
        list of derived frame dicts, or None
    """
    if turn_event is None:
        return None

    frames = run_info['frames']
    t_on = turn_event['t_turn_on_ns']
    t_off = turn_event['t_turn_off_ns']
    pre_ns = args.pre_turn_ms * 1e6
    post_ns = args.post_turn_ms * 1e6

    window_start = t_on - pre_ns
    window_end = t_off + post_ns
    direction = turn_event['turn_dir']
    lr_id = 0 if direction == 'left' else 1
    lr_name = 'Left' if direction == 'left' else 'Right'

    result = []
    for fr in frames:
        t = fr['timestamp_ns']
        if t < window_start or t > window_end:
            continue

        phase = classify_frame_phase(
            t, t_on, t_off, args.pre_turn_ms, args.post_turn_ms, 0)
        t_rel = ns_to_ms(t) - ns_to_ms(t_on)

        result.append({
            'image_name': fr['image_name'],
            'label_id': lr_id,
            'label_name': lr_name,
            'timestamp_ns': t,
            'linear_x': fr['linear_x'],
            'angular_z': fr['angular_z'],
            'valid': fr['valid'],
            'orig_action_id': fr['action_id'],
            'orig_action_name': fr['action_name'],
            'run_name': run_info['run_name'],
            'split': run_info['split'],
            't_rel_ms': round(t_rel, 1),
            'phase': phase,
        })

    return result if result else None


# ============================================================================
# 任务 C: stage4
# ============================================================================
STAGE4_LABELS = {
    'Follow': 0, 'Approach': 1, 'Turn': 2, 'Recover': 3
}


def derive_stage4(run_info, turn_event, args):
    """
    派生四阶段分类数据集。
    Follow / Approach / Turn / Recover。

    Returns:
        list of derived frame dicts, or None
    """
    if turn_event is None:
        return None

    frames = run_info['frames']
    t_on = turn_event['t_turn_on_ns']
    t_off = turn_event['t_turn_off_ns']
    pre_ms = args.pre_turn_ms
    recover_ms = args.recover_ms
    max_follow = args.max_follow_frames
    rng = random.Random(args.seed)

    # 四阶段边界 (ms)
    on_ms = ns_to_ms(t_on)
    off_ms = ns_to_ms(t_off)
    approach_start = on_ms - pre_ms
    turn_start = on_ms - 200
    turn_end = off_ms + 200
    recover_end = off_ms + recover_ms

    buckets = {'Follow': [], 'Approach': [], 'Turn': [], 'Recover': []}

    for fr in frames:
        t_ms = ns_to_ms(fr['timestamp_ns'])
        t_rel = t_ms - on_ms

        if t_ms < approach_start:
            phase = 'Follow'
        elif t_ms < turn_start:
            phase = 'Approach'
        elif t_ms <= turn_end:
            phase = 'Turn'
        elif t_ms <= recover_end:
            phase = 'Recover'
        else:
            continue  # 超出 recover 窗口

        if fr['action_name'] == 'Backward':
            continue

        out = {
            'image_name': fr['image_name'],
            'label_id': STAGE4_LABELS[phase],
            'label_name': phase,
            'timestamp_ns': fr['timestamp_ns'],
            'linear_x': fr['linear_x'],
            'angular_z': fr['angular_z'],
            'valid': fr['valid'],
            'orig_action_id': fr['action_id'],
            'orig_action_name': fr['action_name'],
            'run_name': run_info['run_name'],
            'split': run_info['split'],
            't_rel_ms': round(t_rel, 1),
            'phase': phase,
        }
        buckets[phase].append(out)

    # Follow 限制
    if len(buckets['Follow']) > max_follow:
        rng.shuffle(buckets['Follow'])
        buckets['Follow'] = buckets['Follow'][:max_follow]

    result = []
    for phase in ['Follow', 'Approach', 'Turn', 'Recover']:
        result.extend(buckets[phase])
    result.sort(key=lambda x: x['timestamp_ns'])
    return result if result else None


# ============================================================================
# 统一派生流程
# ============================================================================
TASK_FUNCS = {
    'action3_balanced': derive_action3_balanced,
    'junction_lr': derive_junction_lr,
    'stage4': derive_stage4,
}

TASK_LABEL_NAMES = {
    'action3_balanced': {0: 'Left', 1: 'Straight', 2: 'Right'},
    'junction_lr': {0: 'Left', 1: 'Right'},
    'stage4': {0: 'Follow', 1: 'Approach', 2: 'Turn', 3: 'Recover'},
}


def process_task(task_name, all_runs, turn_events, args):
    """
    对单个任务执行全部 run 的派生。

    Returns:
        stats: dict
    """
    func = TASK_FUNCS[task_name]
    task_dir = os.path.join(args.dst_root, f'{task_name}_v1')
    label_names = TASK_LABEL_NAMES[task_name]

    print(f'\n{"=" * 70}')
    print(f'  任务: {task_name}')
    print(f'  输出: {task_dir}/')
    print(f'{"=" * 70}')

    # 统计
    split_stats = {s: {'runs': 0, 'frames': 0, 'labels': defaultdict(int)}
                   for s in SPLITS}
    junction_stats = defaultdict(int)
    turndir_stats = defaultdict(int)
    run_stats = []
    skipped = []
    manifest_rows = []

    for run in all_runs:
        rn = run['run_name']
        sp = run['split']
        te = turn_events.get(rn)

        derived = func(run, te, args)

        if derived is None or len(derived) == 0:
            reason = '无 turn 事件' if te is None else 'turn 派生为空'
            skipped.append({'run_name': rn, 'split': sp, 'reason': reason})
            print(f'  [跳过] {rn} ({sp}): {reason}')
            continue

        # 写出
        out_run_dir = os.path.join(task_dir, sp, rn)
        if args.force and os.path.exists(out_run_dir):
            shutil.rmtree(out_run_dir)

        src_img_dir = os.path.join(run['run_dir'], 'images')
        n_out = write_derived_run(out_run_dir, derived, src_img_dir,
                                  args.copy_mode)

        # 统计
        split_stats[sp]['runs'] += 1
        split_stats[sp]['frames'] += n_out
        for fr in derived:
            split_stats[sp]['labels'][fr['label_name']] += 1

        junction_stats[run['junction_id']] += n_out
        turndir_stats[run['turn_dir']] += n_out

        run_stats.append({
            'run_name': rn, 'split': sp,
            'junction_id': run['junction_id'],
            'turn_dir': run['turn_dir'],
            'derived_frames': n_out,
            'turn_method': te['method'] if te else None,
        })
        for fr in derived:
            manifest_rows.append({
                'run_name': rn, 'split': sp,
                'image_name': fr['image_name'],
                'label_id': fr['label_id'],
                'label_name': fr['label_name'],
                'phase': fr['phase'],
            })

        ldist = defaultdict(int)
        for fr in derived:
            ldist[fr['label_name']] += 1
        ld_str = ', '.join(f'{k}:{v}' for k, v in sorted(ldist.items()))
        print(f'  [{sp:5s}] {rn}: {n_out} 帧  ({ld_str})')

    # ---- 汇总打印 ----
    print(f'\n  ── 汇总 ──')
    total_runs = 0
    total_frames = 0
    for sp in SPLITS:
        st = split_stats[sp]
        if st['runs'] == 0:
            continue
        total_runs += st['runs']
        total_frames += st['frames']
        ld = ', '.join(f'{k}:{v}' for k, v in sorted(st['labels'].items()))
        print(f'  {sp:5s}: {st["runs"]:3d} runs, {st["frames"]:5d} 帧  ({ld})')
    print(f'  Total: {total_runs:3d} runs, {total_frames:5d} 帧')
    if skipped:
        print(f'  跳过:  {len(skipped)} runs')

    # ---- 保存元数据 ----
    os.makedirs(task_dir, exist_ok=True)

    # dataset_summary.json
    summary = {
        'task': task_name,
        'label_names': label_names,
        'splits': {},
        'junction_distribution': dict(junction_stats),
        'turn_dir_distribution': dict(turndir_stats),
        'per_run': run_stats,
        'skipped_count': len(skipped),
    }
    for sp in SPLITS:
        st = split_stats[sp]
        summary['splits'][sp] = {
            'runs': st['runs'],
            'frames': st['frames'],
            'label_distribution': dict(st['labels']),
        }

    with open(os.path.join(task_dir, 'dataset_summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'  [✓] dataset_summary.json')

    # derive_manifest.csv
    if manifest_rows:
        mpath = os.path.join(task_dir, 'derive_manifest.csv')
        with open(mpath, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=[
                'run_name', 'split', 'image_name',
                'label_id', 'label_name', 'phase'])
            w.writeheader()
            w.writerows(manifest_rows)
        print(f'  [✓] derive_manifest.csv ({len(manifest_rows)} rows)')

    # skipped_runs.csv
    if skipped:
        spath = os.path.join(task_dir, 'skipped_runs.csv')
        with open(spath, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['run_name', 'split', 'reason'])
            w.writeheader()
            w.writerows(skipped)
        print(f'  [✓] skipped_runs.csv ({len(skipped)} runs)')

    print(f'{"=" * 70}')
    return summary


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='走廊导航阶段一数据集派生',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_root', type=str, default='./data/corridor',
                        help='已划分的 corridor 数据根目录')
    parser.add_argument('--dst_root', type=str, default='./data/stage1',
                        help='派生数据集输出根目录')
    parser.add_argument('--task', type=str, default='all',
                        choices=['action3_balanced', 'junction_lr',
                                 'stage4', 'all'],
                        help='要派生的任务')
    parser.add_argument('--valid_only', type=lambda x:
                        str(x).lower() in ('true', '1', 'yes'),
                        default=True, help='丢弃 valid=0 帧')

    # Turn 检测参数
    parser.add_argument('--turn_k_consecutive', type=int, default=3,
                        help='连续 K 帧认定为 turn')
    parser.add_argument('--turn_w_threshold', type=float, default=0.3,
                        help='angular_z 备选检测阈值')

    # 时间窗口参数
    parser.add_argument('--pre_turn_ms', type=float, default=1200,
                        help='turn_on 前的 approach 窗口 (ms)')
    parser.add_argument('--post_turn_ms', type=float, default=1000,
                        help='turn_off 后的 junction_lr 窗口 (ms)')
    parser.add_argument('--recover_ms', type=float, default=1500,
                        help='turn_off 后的 recover 窗口 (ms)')

    # 采样参数
    parser.add_argument('--max_follow_frames', type=int, default=15,
                        help='Follow 阶段最多保留帧数')
    parser.add_argument('--straight_ratio_cap', type=float, default=1.5,
                        help='Straight 最多为 (Left+Right) 的倍数')

    # 输出
    parser.add_argument('--copy_mode', type=str, default='symlink',
                        choices=['copy', 'symlink'],
                        help='图片复制方式')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true',
                        help='覆盖已有输出')

    args = parser.parse_args()

    print('=' * 70)
    print('  走廊导航阶段一数据集派生')
    print('=' * 70)
    print(f'  输入: {os.path.abspath(args.src_root)}')
    print(f'  输出: {os.path.abspath(args.dst_root)}')
    print(f'  任务: {args.task}')
    print(f'  Turn 检测: K={args.turn_k_consecutive}, '
          f'ω阈值={args.turn_w_threshold}')
    print(f'  时间窗口: pre={args.pre_turn_ms}ms, post={args.post_turn_ms}ms, '
          f'recover={args.recover_ms}ms')
    print(f'  Follow 上限: {args.max_follow_frames} 帧')
    print(f'  Straight 比例上限: {args.straight_ratio_cap}x')
    print('=' * 70)

    # ---- [1] 扫描 ----
    print('\n[1/3] 扫描 runs...')
    all_runs = scan_all_runs(args.src_root, valid_only=args.valid_only)
    if not all_runs:
        print('  ✗ 未找到任何有效 run!')
        sys.exit(1)

    split_counts = defaultdict(int)
    for r in all_runs:
        split_counts[r['split']] += 1
    for sp, cnt in sorted(split_counts.items()):
        print(f'  {sp}: {cnt} runs')
    print(f'  总计: {len(all_runs)} runs, '
          f'{sum(len(r["frames"]) for r in all_runs)} 帧')

    # ---- [2] Turn 检测 ----
    print('\n[2/3] 检测 turn 事件...')
    turn_events = {}
    no_turn = []
    for run in all_runs:
        te = detect_turn_event(
            run['frames'],
            k=args.turn_k_consecutive,
            w_threshold=args.turn_w_threshold)
        if te:
            turn_events[run['run_name']] = te
            dur_ms = ns_to_ms(te['t_turn_off_ns'] - te['t_turn_on_ns'])
            print(f'  {run["run_name"]}: {te["turn_dir"]:5s} '
                  f'[{te["method"]}] '
                  f'idx=[{te["idx_on"]},{te["idx_off"]}] '
                  f'dur={dur_ms:.0f}ms')
        else:
            no_turn.append(run['run_name'])
            print(f'  {run["run_name"]}: ✗ 未检测到 turn')

    print(f'\n  检测到 turn: {len(turn_events)}/{len(all_runs)}')
    if no_turn:
        print(f'  未检测到: {no_turn}')

    # ---- [3] 派生 ----
    print('\n[3/3] 派生任务数据集...')
    tasks = list(TASK_FUNCS.keys()) if args.task == 'all' else [args.task]
    all_summaries = {}

    for task_name in tasks:
        summary = process_task(task_name, all_runs, turn_events, args)
        all_summaries[task_name] = summary

    # ---- 总结 ----
    print('\n' + '=' * 70)
    print('  全部完成!')
    print('=' * 70)
    for tn, sm in all_summaries.items():
        total = sum(s['frames'] for s in sm['splits'].values())
        runs = sum(s['runs'] for s in sm['splits'].values())
        print(f'  {tn + "_v1":25s}  {runs:3d} runs  {total:5d} 帧  '
              f'(跳过 {sm["skipped_count"]})')
    print(f'\n  输出目录: {os.path.abspath(args.dst_root)}/')
    print('=' * 70)


if __name__ == '__main__':
    main()
