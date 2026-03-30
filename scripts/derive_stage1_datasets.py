"""
走廊导航阶段一数据集派生脚本
================================
从已经 downsample 并完成 train/val/test 划分的 corridor 数据中,
派生四个任务数据集:

  A) action3_balanced_v1  — 三分类 (Left / Straight / Right)
  B) junction_lr_v1       — 二分类 (Left / Right), 仅转弯窗口
  C) stage3_v1            — 三分类 (Approach / Turn / Recover)
  D) stage4_v1            — 四分类 (Follow / Approach / Turn / Recover)

重要假设:
  输入数据已经经过 downsample_corridor.py 智能降采样,
  本脚本不再做第二轮激进降采样, 仅做任务派生和必要的轻量裁剪。

用法:
  # 派生全部任务
  python scripts/derive_stage1_datasets.py --src_root ./data/corridor --dst_root ./data/stage1 --task all --force

  # 仅派生 action3_balanced
  python scripts/derive_stage1_datasets.py --src_root ./data/corridor --dst_root ./data/stage1 --task action3_balanced

  # 仅派生 stage3
  python scripts/derive_stage1_datasets.py --src_root ./data/corridor --dst_root ./data/stage1 --task stage3 --force

  # 自定义参数
  python scripts/derive_stage1_datasets.py --src_root ./data/corridor --dst_root ./data/stage1 --task all --pre_turn_ms 2000 --recover_ms 1800 --force
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

ORIG_ACTIONS = {0: 'Forward', 1: 'Backward', 2: 'Left', 3: 'Right', 4: 'Stop'}

# 目录名解析: J1_left_r03
_RE_RUN = re.compile(
    r'^J(?P<jid>\d+)_(?P<turn>left|right)_r(?P<rep>\d+)$', re.IGNORECASE)


# ============================================================================
# 数据读取
# ============================================================================

def load_run_labels(run_dir, valid_only=True):
    """
    读取 run_dir/labels.csv 的帧列表。

    Returns:
        list of dict, 含 image_name, action_id, action_name,
            timestamp_ns, linear_x, angular_z, time_diff_ms, valid
    """
    csv_path = os.path.join(run_dir, 'labels.csv')
    if not os.path.isfile(csv_path):
        return []

    frames = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
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
                'time_diff_ms': float(row.get('time_diff_ms', '0')),
                'valid': valid,
            })
    return frames


def scan_all_runs(src_root, valid_only=True):
    """
    扫描 src_root/{train,val,test} 下全部有效 run。

    Returns:
        list of dict: [{run_name, run_dir, split, frames,
                         junction_id, turn_dir, rep_id}, ...]
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
            if not os.path.isdir(os.path.join(rd, 'images')):
                continue

            frames = load_run_labels(rd, valid_only)
            if not frames:
                continue

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

    策略 (按优先级):
      1. action_name 连续 K 帧为 Left 或 Right → t_turn_on / t_turn_off
      2. |angular_z| >= threshold 且符号一致连续 K 帧

    Returns:
        dict: {turn_dir, t_turn_on_ns, t_turn_off_ns,
               idx_on, idx_off, method}
        或 None (无法检测)
    """
    n = len(frames)
    if n < k:
        return None

    # ---- 方法1: action_name 连续 K 帧 ----
    for i in range(n - k + 1):
        names = [frames[j]['action_name'] for j in range(i, i + k)]
        if all(nm in ('Left', 'Right') for nm in names) and \
                len(set(names)) == 1:
            direction = names[0].lower()
            # 向后扩展 turn 结束
            idx_on = i
            idx_off = i + k - 1
            while idx_off + 1 < n and \
                    frames[idx_off + 1]['action_name'].lower() == direction:
                idx_off += 1
            return {
                'turn_dir': direction,
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
                return {
                    'turn_dir': direction,
                    't_turn_on_ns': frames[idx_on]['timestamp_ns'],
                    't_turn_off_ns': frames[idx_off]['timestamp_ns'],
                    'idx_on': idx_on,
                    'idx_off': idx_off,
                    'method': 'angular_z',
                }

    return None


def ns_to_ms(ns):
    """纳秒转毫秒"""
    return ns / 1e6


# ============================================================================
# 帧分阶段
# ============================================================================

def classify_phase_action3(t_ns, t_on_ns, t_off_ns, pre_ms, recover_ms):
    """action3_balanced 用: Follow / Approach / Turn / Recover / Post"""
    t = ns_to_ms(t_ns)
    on = ns_to_ms(t_on_ns)
    off = ns_to_ms(t_off_ns)

    if t < on - pre_ms:
        return 'Follow'
    elif t < on:
        return 'Approach'
    elif t <= off:
        return 'Turn'
    elif t <= off + recover_ms:
        return 'Recover'
    else:
        return 'Post'


def classify_phase_stage4(t_ns, t_on_ns, t_off_ns,
                          pre_ms, recover_ms, margin_ms=300):
    """
    stage4 四阶段分类:
      Follow:   t < t_on - pre_ms
      Approach: [t_on - pre_ms, t_on - margin_ms)
      Turn:     [t_on - margin_ms, t_off + margin_ms]
      Recover:  (t_off + margin_ms, t_off + recover_ms]
    超出范围返回 None
    """
    t = ns_to_ms(t_ns)
    on = ns_to_ms(t_on_ns)
    off = ns_to_ms(t_off_ns)

    if t < on - pre_ms:
        return 'Follow'
    elif t < on - margin_ms:
        return 'Approach'
    elif t <= off + margin_ms:
        return 'Turn'
    elif t <= off + recover_ms:
        return 'Recover'
    else:
        return None


# ============================================================================
# 文件操作
# ============================================================================

def copy_image(src, dst, mode='copy'):
    """复制或链接单张图片"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if mode == 'symlink':
        src_abs = os.path.abspath(src)
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src_abs, dst)
    else:
        shutil.copy2(src, dst)


def write_derived_run(out_dir, frames_out, src_img_dir, copy_mode):
    """
    写出一个派生 run: images/ + labels.csv + meta.json。

    Returns:
        int: 写出帧数
    """
    img_out = os.path.join(out_dir, 'images')
    os.makedirs(img_out, exist_ok=True)

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
        'run_name', 'split', 't_rel_ms', 'phase',
    ]
    with open(os.path.join(out_dir, 'labels.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fr in frames_out:
            w.writerow({k: fr.get(k, '') for k in fieldnames})

    # meta.json
    label_dist = defaultdict(int)
    for fr in frames_out:
        label_dist[fr['label_name']] += 1
    meta = {
        'total_frames': len(frames_out),
        'valid_frames': len(frames_out),
        'label_distribution': dict(label_dist),
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return len(frames_out)


# ============================================================================
# 任务 A: action3_balanced_v1
# ============================================================================

ACTION3_MAP = {
    'Left':    (0, 'Left'),
    'Right':   (2, 'Right'),
    'Forward': (1, 'Straight'),
    'Stop':    (1, 'Straight'),
    # Backward → 丢弃
}


def derive_action3_balanced(run_info, turn_event, args):
    """
    三分类派生。
    输入数据已经 downsample, 默认保留全部样本。
    仅当 Straight/min(Left,Right) > ratio_cap 时轻量裁剪最远 Follow 帧。

    Returns:
        (frames_out, trim_info) 或 (None, reason_str)
    """
    frames = run_info['frames']
    if turn_event is None:
        return None, '无 turn 事件'

    t_on = turn_event['t_turn_on_ns']
    t_off = turn_event['t_turn_off_ns']

    left_frames = []
    right_frames = []
    straight_by_phase = defaultdict(list)  # phase -> [frame]

    for fr in frames:
        act = fr['action_name']
        if act == 'Backward':
            continue
        if act not in ACTION3_MAP:
            continue

        label_id, label_name = ACTION3_MAP[act]
        phase = classify_phase_action3(
            fr['timestamp_ns'], t_on, t_off,
            args.pre_turn_ms, args.recover_ms)
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

        if act == 'Left':
            left_frames.append(out)
        elif act == 'Right':
            right_frames.append(out)
        else:
            straight_by_phase[phase].append(out)

    # 合并 Straight
    all_straight = []
    for phase in ['Follow', 'Approach', 'Turn', 'Recover', 'Post']:
        all_straight.extend(straight_by_phase.get(phase, []))

    n_left = len(left_frames)
    n_right = len(right_frames)
    n_lr_min = min(n_left, n_right) if n_left > 0 and n_right > 0 else max(n_left, n_right)
    n_straight_before = len(all_straight)

    # 轻量裁剪: 仅当 Straight 超过 ratio_cap * min(Left, Right)
    trim_info = {'trimmed': False, 'before': n_straight_before, 'after': n_straight_before}
    if n_lr_min > 0:
        max_straight = int(args.straight_ratio_cap * n_lr_min)
        if n_straight_before > max_straight:
            # 只裁剪 Follow 帧, 保留 Approach / Turn / Recover / Post 不动
            non_follow = [f for f in all_straight if f['phase'] != 'Follow']
            follow_only = [f for f in all_straight if f['phase'] == 'Follow']
            budget = max_straight - len(non_follow)
            if budget > 0 and len(follow_only) > budget:
                # 保留最靠近 Approach 的 Follow 帧 (按时间降序取最晚的)
                follow_only.sort(key=lambda x: x['timestamp_ns'], reverse=True)
                follow_only = follow_only[:budget]
            elif budget <= 0:
                follow_only = []
            all_straight = non_follow + follow_only
            trim_info = {
                'trimmed': True,
                'before': n_straight_before,
                'after': len(all_straight),
                'budget': max_straight,
            }

    result = left_frames + right_frames + all_straight
    result.sort(key=lambda x: x['timestamp_ns'])
    return result, trim_info


# ============================================================================
# 任务 B: junction_lr_v1
# ============================================================================

def derive_junction_lr(run_info, turn_event, args):
    """
    二分类派生: 仅保留转弯窗口 [t_on - pre_ms, t_off + post_ms]。
    所有帧标签统一继承转向方向。

    Returns:
        (frames_out, info_dict) 或 (None, reason_str)
    """
    if turn_event is None:
        return None, '无 turn 事件'

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
        if fr['action_name'] == 'Backward':
            continue

        t_rel = ns_to_ms(t) - ns_to_ms(t_on)
        # 确定子阶段 (仅信息性, 不影响标签)
        t_ms = ns_to_ms(t)
        on_ms = ns_to_ms(t_on)
        off_ms = ns_to_ms(t_off)
        if t_ms < on_ms:
            phase = 'Pre'
        elif t_ms <= off_ms:
            phase = 'Turn'
        else:
            phase = 'Post'

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

    if not result:
        return None, '转弯窗口内无帧'
    return result, {'window_ms': ns_to_ms(t_off - t_on + pre_ns + post_ns)}


# ============================================================================
# 任务 C: stage4_v1
# ============================================================================

STAGE4_LABELS = {'Follow': 0, 'Approach': 1, 'Turn': 2, 'Recover': 3}


def _sample_phase(frames, max_frames, policy='tail'):
    """
    按策略对单阶段帧列表做裁剪/抽样。

    policy:
      tail:         保留尾部 (最靠近下一阶段)
      uniform:      均匀抽样
      uniform_tail: 前半均匀 + 后半全保留

    Returns:
        list: 裁剪后的帧列表 (保持时间顺序)
    """
    if not frames or max_frames <= 0:
        return frames
    if len(frames) <= max_frames:
        return frames

    # 按时间排序
    frames_sorted = sorted(frames, key=lambda x: x['timestamp_ns'])
    n = len(frames_sorted)

    if policy == 'tail':
        return frames_sorted[n - max_frames:]

    elif policy == 'uniform':
        indices = [round(i * (n - 1) / (max_frames - 1))
                   for i in range(max_frames)] if max_frames > 1 else [n - 1]
        return [frames_sorted[i] for i in indices]

    elif policy == 'uniform_tail':
        # 后半部分全保留, 前半均匀
        tail_n = max_frames // 2
        head_n = max_frames - tail_n
        tail_part = frames_sorted[n - tail_n:]
        head_pool = frames_sorted[:n - tail_n]
        if head_pool and head_n > 0:
            step = max(1, len(head_pool) // head_n)
            head_part = head_pool[::step][:head_n]
        else:
            head_part = []
        return head_part + tail_part

    else:
        # fallback: tail
        return frames_sorted[n - max_frames:]


def derive_stage4(run_info, turn_event, args):
    """
    四阶段分类派生 (v2).

    每阶段分别收集, 按 stage4_max_xxx_frames 裁剪,
    裁剪策略由 stage4_sample_policy 控制.

    Returns:
        (frames_out, info_dict) 或 (None, reason_str)
    """
    if turn_event is None:
        return None, '无 turn 事件'

    frames = run_info['frames']
    t_on = turn_event['t_turn_on_ns']
    t_off = turn_event['t_turn_off_ns']

    # v2 参数 (兼容旧版: 若无 stage4_ 参数则 fallback)
    pre_ms = getattr(args, 'stage4_pre_turn_ms', None) or args.pre_turn_ms
    recover_ms = getattr(args, 'stage4_recover_ms', None) or args.recover_ms
    margin_ms = getattr(args, 'stage4_turn_margin_ms', 300)
    max_follow = getattr(args, 'stage4_max_follow_frames',
                         args.max_follow_frames)
    max_approach = getattr(args, 'stage4_max_approach_frames', 0)
    max_turn = getattr(args, 'stage4_max_turn_frames', 0)
    max_recover = getattr(args, 'stage4_max_recover_frames', 0)
    policy = getattr(args, 'stage4_sample_policy', 'tail')
    drop_no_follow = getattr(args, 'stage4_drop_runs_without_follow', False)
    min_follow = getattr(args, 'stage4_min_follow_frames', 0)

    buckets = {'Follow': [], 'Approach': [], 'Turn': [], 'Recover': []}

    for fr in frames:
        if fr['action_name'] == 'Backward':
            continue

        phase = classify_phase_stage4(
            fr['timestamp_ns'], t_on, t_off,
            pre_ms, recover_ms, margin_ms)
        if phase is None:
            continue

        t_rel = ns_to_ms(fr['timestamp_ns']) - ns_to_ms(t_on)

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

    # 记录裁剪前数量
    raw_counts = {k: len(v) for k, v in buckets.items()}

    # drop_runs_without_follow 检查
    if drop_no_follow and raw_counts['Follow'] == 0:
        return None, '无 Follow 帧 (已启用 drop_runs_without_follow)'

    # min_follow 检查
    if min_follow > 0 and raw_counts['Follow'] < min_follow:
        return None, (f'Follow 帧不足: '
                      f'{raw_counts["Follow"]} < {min_follow}')

    # 按阶段裁剪
    phase_policies = {
        'Follow': ('tail', max_follow),      # Follow 默认 tail
        'Approach': (policy, max_approach),
        'Turn': ('uniform', max_turn),       # Turn 默认 uniform
        'Recover': ('uniform', max_recover), # Recover 默认 uniform
    }

    for phase, (ph_policy, ph_max) in phase_policies.items():
        if ph_max > 0 and len(buckets[phase]) > ph_max:
            buckets[phase] = _sample_phase(
                buckets[phase], ph_max, ph_policy)

    # 裁剪后数量
    derived_counts = {k: len(v) for k, v in buckets.items()}

    result = []
    for phase in ['Follow', 'Approach', 'Turn', 'Recover']:
        result.extend(buckets[phase])
    result.sort(key=lambda x: x['timestamp_ns'])

    if not result:
        return None, '所有阶段均为空'

    info = {
        'raw_counts': raw_counts,
        'derived_counts': derived_counts,
        **derived_counts,
    }
    return result, info


# ============================================================================
# Stage3: 三阶段 (Approach / Turn / Recover)
# ============================================================================

STAGE3_LABELS = {'Approach': 0, 'Turn': 1, 'Recover': 2}


def derive_stage3(run_info, turn_event, args):
    """
    三阶段分类派生 — 丢弃 Follow, 只保留 Approach/Turn/Recover.

    复用 classify_phase_stage4() 做阶段判定.

    Returns:
        (frames_out, info_dict) 或 (None, reason_str)
    """
    if turn_event is None:
        return None, '无 turn 事件'

    frames = run_info['frames']
    t_on = turn_event['t_turn_on_ns']
    t_off = turn_event['t_turn_off_ns']

    # stage3 专用参数 (fallback 到通用参数)
    pre_ms = getattr(args, 'stage3_pre_turn_ms', None) or args.pre_turn_ms
    recover_ms = getattr(args, 'stage3_recover_ms', None) or args.recover_ms
    margin_ms = getattr(args, 'stage3_turn_margin_ms', 300)
    max_approach = getattr(args, 'stage3_max_approach_frames', 0)
    max_turn = getattr(args, 'stage3_max_turn_frames', 0)
    max_recover = getattr(args, 'stage3_max_recover_frames', 0)
    policy = getattr(args, 'stage3_sample_policy', 'tail')

    buckets = {'Approach': [], 'Turn': [], 'Recover': []}

    for fr in frames:
        if fr['action_name'] == 'Backward':
            continue

        phase = classify_phase_stage4(
            fr['timestamp_ns'], t_on, t_off,
            pre_ms, recover_ms, margin_ms)

        # 丢弃 Follow 和超出范围
        if phase is None or phase == 'Follow':
            continue

        t_rel = ns_to_ms(fr['timestamp_ns']) - ns_to_ms(t_on)

        out = {
            'image_name': fr['image_name'],
            'label_id': STAGE3_LABELS[phase],
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

    # 裁剪前数量
    raw_counts = {k: len(v) for k, v in buckets.items()}

    # 按阶段裁剪
    phase_policies = {
        'Approach': (policy, max_approach),      # 默认 tail, 保留靠近 Turn 的
        'Turn': ('uniform', max_turn),           # 均匀采样
        'Recover': ('uniform', max_recover),     # 均匀采样
    }

    for phase, (ph_policy, ph_max) in phase_policies.items():
        if ph_max > 0 and len(buckets[phase]) > ph_max:
            buckets[phase] = _sample_phase(
                buckets[phase], ph_max, ph_policy)

    # 裁剪后数量
    derived_counts = {k: len(v) for k, v in buckets.items()}

    result = []
    for phase in ['Approach', 'Turn', 'Recover']:
        result.extend(buckets[phase])
    result.sort(key=lambda x: x['timestamp_ns'])

    if not result:
        return None, '所有阶段均为空'

    info = {
        'raw_counts': raw_counts,
        'derived_counts': derived_counts,
        **derived_counts,
    }
    return result, info


# ============================================================================
# 统一派生调度
# ============================================================================

TASK_FUNCS = {
    'action3_balanced': derive_action3_balanced,
    'junction_lr': derive_junction_lr,
    'stage3': derive_stage3,
    'stage4': derive_stage4,
}

TASK_LABEL_NAMES = {
    'action3_balanced': {0: 'Left', 1: 'Straight', 2: 'Right'},
    'junction_lr': {0: 'Left', 1: 'Right'},
    'stage3': {0: 'Approach', 1: 'Turn', 2: 'Recover'},
    'stage4': {0: 'Follow', 1: 'Approach', 2: 'Turn', 3: 'Recover'},
}


def process_task(task_name, all_runs, turn_events, args):
    """
    对单个任务执行全部 run 的派生。

    Returns:
        summary dict
    """
    func = TASK_FUNCS[task_name]
    task_dir = os.path.join(args.dst_root, f'{task_name}_v1')
    label_names = TASK_LABEL_NAMES[task_name]

    print(f'\n{"=" * 70}')
    print(f'  任务: {task_name}')
    print(f'  输出: {task_dir}/')
    print(f'  类别: {label_names}')
    print(f'{"=" * 70}')

    split_stats = {s: {'runs': 0, 'frames': 0, 'labels': defaultdict(int)}
                   for s in SPLITS}
    run_details = []
    skipped = []
    manifest_rows = []
    trim_log = []  # action3 专用

    for run in all_runs:
        rn = run['run_name']
        sp = run['split']
        te = turn_events.get(rn)

        result, info = func(run, te, args)

        if result is None or len(result) == 0:
            reason = info if isinstance(info, str) else '派生结果为空'
            skipped.append({'run_name': rn, 'split': sp, 'reason': reason})
            print(f'  [跳过] {rn} ({sp}): {reason}')
            continue

        # 写出
        out_run_dir = os.path.join(task_dir, sp, rn)
        if args.force and os.path.exists(out_run_dir):
            shutil.rmtree(out_run_dir)

        src_img_dir = os.path.join(run['run_dir'], 'images')
        n_out = write_derived_run(out_run_dir, result, src_img_dir,
                                  args.copy_mode)

        # 统计
        split_stats[sp]['runs'] += 1
        split_stats[sp]['frames'] += n_out
        for fr in result:
            split_stats[sp]['labels'][fr['label_name']] += 1

        # 标签分布字符串
        ldist = defaultdict(int)
        for fr in result:
            ldist[fr['label_name']] += 1
        ld_str = ', '.join(f'{k}:{v}' for k, v in sorted(ldist.items()))

        rd = {
            'run_name': rn, 'split': sp,
            'junction_id': run['junction_id'],
            'turn_dir': run['turn_dir'],
            'derived_frames': n_out,
            'label_distribution': dict(ldist),
        }
        if te:
            rd['turn_method'] = te['method']

        # action3 裁剪信息
        if task_name == 'action3_balanced' and isinstance(info, dict):
            rd['trim_info'] = info
            if info.get('trimmed'):
                trim_log.append(f'    {rn}: {info["before"]} → {info["after"]}')

        # stage3/stage4: 记录每 run 原始/派生阶段长度
        if task_name in ('stage3', 'stage4') and isinstance(info, dict):
            rd['raw_counts'] = info.get('raw_counts', {})
            rd['derived_counts'] = info.get('derived_counts', {})

        run_details.append(rd)

        for fr in result:
            manifest_rows.append({
                'run_name': rn, 'split': sp,
                'image_name': fr['image_name'],
                'label_id': fr['label_id'],
                'label_name': fr['label_name'],
                'phase': fr.get('phase', ''),
            })

        print(f'  [{sp:5s}] {rn}: {n_out:4d} 帧  ({ld_str})')

    # ---- 汇总 ----
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
    if trim_log:
        print(f'\n  Straight 轻量裁剪:')
        for tl in trim_log:
            print(tl)

    # ---- 保存元数据 ----
    os.makedirs(task_dir, exist_ok=True)

    # dataset_summary.json
    summary = {
        'task': task_name,
        'version': 'v1',
        'label_names': {str(k): v for k, v in label_names.items()},
        'params': {
            'pre_turn_ms': args.pre_turn_ms,
            'post_turn_ms': args.post_turn_ms,
            'recover_ms': args.recover_ms,
            'max_follow_frames': args.max_follow_frames,
            'straight_ratio_cap': args.straight_ratio_cap,
            'turn_k_consecutive': args.turn_k_consecutive,
            'turn_w_threshold': args.turn_w_threshold,
            'seed': args.seed,
        },
        'splits': {},
        'per_run': run_details,
        'skipped_count': len(skipped),
        'skipped_runs': skipped,
    }
    for sp in SPLITS:
        st = split_stats[sp]
        summary['splits'][sp] = {
            'runs': st['runs'],
            'frames': st['frames'],
            'label_distribution': dict(st['labels']),
        }

    # action3 额外统计: 裁剪前后比例
    if task_name == 'action3_balanced':
        trim_summary = {
            'total_trimmed_runs': sum(
                1 for rd in run_details
                if rd.get('trim_info', {}).get('trimmed', False)),
            'total_straight_before': sum(
                rd.get('trim_info', {}).get('before', 0)
                for rd in run_details),
            'total_straight_after': sum(
                rd.get('trim_info', {}).get('after',
                    rd.get('trim_info', {}).get('before', 0))
                for rd in run_details),
        }
        summary['straight_trim_summary'] = trim_summary

    # stage3 额外统计
    if task_name == 'stage3':
        s3_params = {
            'stage3_pre_turn_ms': getattr(
                args, 'stage3_pre_turn_ms', None) or args.pre_turn_ms,
            'stage3_recover_ms': getattr(
                args, 'stage3_recover_ms', None) or args.recover_ms,
            'stage3_turn_margin_ms': getattr(
                args, 'stage3_turn_margin_ms', 300),
            'stage3_max_approach_frames': getattr(
                args, 'stage3_max_approach_frames', 0),
            'stage3_max_turn_frames': getattr(
                args, 'stage3_max_turn_frames', 0),
            'stage3_max_recover_frames': getattr(
                args, 'stage3_max_recover_frames', 0),
            'stage3_sample_policy': getattr(
                args, 'stage3_sample_policy', 'tail'),
        }
        summary['stage3'] = s3_params

        # 每阶段裁剪前后汇总
        phase_trim = {'Approach': [0, 0], 'Turn': [0, 0],
                      'Recover': [0, 0]}
        for rd in run_details:
            raw = rd.get('raw_counts', {})
            derived = rd.get('derived_counts', {})
            for ph in phase_trim:
                phase_trim[ph][0] += raw.get(ph, 0)
                phase_trim[ph][1] += derived.get(ph, 0)
        summary['stage3']['phase_trim_summary'] = {
            ph: {'raw': v[0], 'derived': v[1]}
            for ph, v in phase_trim.items()
        }

    # stage4 v2 额外统计
    if task_name == 'stage4':
        s4_params = {
            'stage4_pre_turn_ms': getattr(
                args, 'stage4_pre_turn_ms', None) or args.pre_turn_ms,
            'stage4_recover_ms': getattr(
                args, 'stage4_recover_ms', None) or args.recover_ms,
            'stage4_turn_margin_ms': getattr(
                args, 'stage4_turn_margin_ms', 300),
            'stage4_max_follow_frames': getattr(
                args, 'stage4_max_follow_frames', args.max_follow_frames),
            'stage4_max_approach_frames': getattr(
                args, 'stage4_max_approach_frames', 0),
            'stage4_max_turn_frames': getattr(
                args, 'stage4_max_turn_frames', 0),
            'stage4_max_recover_frames': getattr(
                args, 'stage4_max_recover_frames', 0),
            'stage4_sample_policy': getattr(
                args, 'stage4_sample_policy', 'tail'),
            'stage4_drop_runs_without_follow': getattr(
                args, 'stage4_drop_runs_without_follow', False),
            'stage4_min_follow_frames': getattr(
                args, 'stage4_min_follow_frames', 0),
        }
        summary['stage4_v2'] = s4_params

        # 每阶段裁剪前后汇总
        phase_trim = {'Follow': [0, 0], 'Approach': [0, 0],
                      'Turn': [0, 0], 'Recover': [0, 0]}
        for rd in run_details:
            raw = rd.get('raw_counts', {})
            derived = rd.get('derived_counts', {})
            for ph in phase_trim:
                phase_trim[ph][0] += raw.get(ph, 0)
                phase_trim[ph][1] += derived.get(ph, 0)
        summary['stage4_v2']['phase_trim_summary'] = {
            ph: {'raw': v[0], 'derived': v[1]}
            for ph, v in phase_trim.items()
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
        description='走廊导航阶段一数据集派生 '
                    '(输入: 已降采样+已划分的 corridor 数据)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_root', type=str, default='./data/corridor',
                        help='已划分的走廊数据根目录 (含 train/val/test/)')
    parser.add_argument('--dst_root', type=str, default='./data/stage1',
                        help='派生输出根目录')
    parser.add_argument('--task', type=str, default='all',
                        choices=['action3_balanced', 'junction_lr',
                                 'stage3', 'stage4', 'all'],
                        help='要派生的任务')
    parser.add_argument('--valid_only', type=lambda x:
                        str(x).lower() in ('true', '1', 'yes'),
                        default=True, help='丢弃 valid=0 帧')

    # Turn 检测
    parser.add_argument('--turn_k_consecutive', type=int, default=3,
                        help='连续 K 帧判定 turn')
    parser.add_argument('--turn_w_threshold', type=float, default=0.3,
                        help='angular_z 备选阈值')

    # 时间窗口
    parser.add_argument('--pre_turn_ms', type=float, default=2000,
                        help='turn_on 前的窗口 (ms)')
    parser.add_argument('--post_turn_ms', type=float, default=1200,
                        help='junction_lr: turn_off 后窗口 (ms)')
    parser.add_argument('--recover_ms', type=float, default=1800,
                        help='stage4/action3: turn_off 后 recover 窗口 (ms)')

    # 采样
    parser.add_argument('--max_follow_frames', type=int, default=20,
                        help='Follow 阶段最多保留帧数')
    parser.add_argument('--straight_ratio_cap', type=float, default=1.8,
                        help='action3: Straight / min(Left,Right) 上限')

    # Stage3 专用
    s3 = parser.add_argument_group('Stage3 参数')
    s3.add_argument('--stage3_pre_turn_ms', type=float, default=None,
                    help='stage3 专用 pre_turn_ms (默认沿用 --pre_turn_ms)')
    s3.add_argument('--stage3_recover_ms', type=float, default=None,
                    help='stage3 专用 recover_ms (默认沿用 --recover_ms)')
    s3.add_argument('--stage3_turn_margin_ms', type=float, default=300,
                    help='Turn 边界前后 margin (ms)')
    s3.add_argument('--stage3_max_approach_frames', type=int, default=0,
                    help='Approach 最多帧 (0=不限)')
    s3.add_argument('--stage3_max_turn_frames', type=int, default=0,
                    help='Turn 最多帧 (0=不限)')
    s3.add_argument('--stage3_max_recover_frames', type=int, default=0,
                    help='Recover 最多帧 (0=不限)')
    s3.add_argument('--stage3_sample_policy', type=str, default='tail',
                    choices=['tail', 'uniform', 'uniform_tail'],
                    help='阶段裁剪策略')

    # Stage4 v2 专用
    s4 = parser.add_argument_group('Stage4 v2 参数')
    s4.add_argument('--stage4_pre_turn_ms', type=float, default=None,
                    help='stage4 专用 pre_turn_ms (默认沿用 --pre_turn_ms)')
    s4.add_argument('--stage4_recover_ms', type=float, default=None,
                    help='stage4 专用 recover_ms (默认沿用 --recover_ms)')
    s4.add_argument('--stage4_turn_margin_ms', type=float, default=300,
                    help='Turn 边界前后 margin (ms)')
    s4.add_argument('--stage4_max_follow_frames', type=int, default=None,
                    help='Follow 最多帧 (默认沿用 --max_follow_frames)')
    s4.add_argument('--stage4_max_approach_frames', type=int, default=0,
                    help='Approach 最多帧 (0=不限)')
    s4.add_argument('--stage4_max_turn_frames', type=int, default=0,
                    help='Turn 最多帧 (0=不限)')
    s4.add_argument('--stage4_max_recover_frames', type=int, default=0,
                    help='Recover 最多帧 (0=不限)')
    s4.add_argument('--stage4_sample_policy', type=str, default='tail',
                    choices=['tail', 'uniform', 'uniform_tail'],
                    help='阶段裁剪策略')
    s4.add_argument('--stage4_drop_runs_without_follow',
                    action='store_true',
                    help='跳过无 Follow 的 run')
    s4.add_argument('--stage4_min_follow_frames', type=int, default=0,
                    help='Follow 最少帧数 (不足则跳过)')

    # 输出
    parser.add_argument('--copy_mode', type=str, default='copy',
                        choices=['copy', 'symlink'],
                        help='图片复制方式 (copy 优先)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true',
                        help='覆盖已有输出目录')

    args = parser.parse_args()

    print('=' * 70)
    print('  走廊导航阶段一数据集派生')
    print('=' * 70)
    print(f'  输入:   {os.path.abspath(args.src_root)}')
    print(f'  输出:   {os.path.abspath(args.dst_root)}')
    print(f'  任务:   {args.task}')
    print(f'  Turn:   K={args.turn_k_consecutive}, '
          f'w阈值={args.turn_w_threshold}')
    print(f'  窗口:   pre={args.pre_turn_ms}ms, '
          f'post={args.post_turn_ms}ms, '
          f'recover={args.recover_ms}ms')
    print(f'  Follow: 最多 {args.max_follow_frames} 帧')
    print(f'  Straight 上限: {args.straight_ratio_cap}x × min(L,R)')
    # stage3 参数
    if args.task in ('stage3', 'all'):
        s3_pre = args.stage3_pre_turn_ms or args.pre_turn_ms
        s3_rec = args.stage3_recover_ms or args.recover_ms
        print(f'  ── Stage3 ──')
        print(f'  S3 pre_turn:     {s3_pre}ms')
        print(f'  S3 recover:      {s3_rec}ms')
        print(f'  S3 margin:       {args.stage3_turn_margin_ms}ms')
        print(f'  S3 max_approach: {args.stage3_max_approach_frames}')
        print(f'  S3 max_turn:     {args.stage3_max_turn_frames}')
        print(f'  S3 max_recover:  {args.stage3_max_recover_frames}')
        print(f'  S3 policy:       {args.stage3_sample_policy}')
    # stage4 v2 参数
    if args.task in ('stage4', 'all'):
        s4_pre = args.stage4_pre_turn_ms or args.pre_turn_ms
        s4_rec = args.stage4_recover_ms or args.recover_ms
        s4_fol = args.stage4_max_follow_frames or args.max_follow_frames
        print(f'  ── Stage4 v2 ──')
        print(f'  S4 pre_turn:  {s4_pre}ms')
        print(f'  S4 recover:   {s4_rec}ms')
        print(f'  S4 margin:    {args.stage4_turn_margin_ms}ms')
        print(f'  S4 max_follow:   {s4_fol}')
        print(f'  S4 max_approach: {args.stage4_max_approach_frames}')
        print(f'  S4 max_turn:     {args.stage4_max_turn_frames}')
        print(f'  S4 max_recover:  {args.stage4_max_recover_frames}')
        print(f'  S4 policy:       {args.stage4_sample_policy}')
        print(f'  S4 drop_no_fol:  {args.stage4_drop_runs_without_follow}')
        print(f'  S4 min_follow:   {args.stage4_min_follow_frames}')
    print(f'  注意:   输入源已降采样, 不做二次激进抽样')
    print('=' * 70)

    # [1] 扫描
    print('\n[1/3] 扫描 runs...')
    all_runs = scan_all_runs(args.src_root, valid_only=args.valid_only)
    if not all_runs:
        print('  ✗ 未找到任何有效 run!')
        sys.exit(1)

    split_counts = defaultdict(int)
    total_frames = 0
    for r in all_runs:
        split_counts[r['split']] += 1
        total_frames += len(r['frames'])
    for sp, cnt in sorted(split_counts.items()):
        print(f'  {sp}: {cnt} runs')
    print(f'  总计: {len(all_runs)} runs, {total_frames} 帧')

    # [2] Turn 检测
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
            print(f'  {run["run_name"]:20s}  {te["turn_dir"]:5s}  '
                  f'[{te["method"]:12s}]  '
                  f'idx=[{te["idx_on"]:3d},{te["idx_off"]:3d}]  '
                  f'dur={dur_ms:6.0f}ms')
        else:
            no_turn.append(run['run_name'])
            print(f'  {run["run_name"]:20s}  ✗ 未检测到 turn')

    print(f'\n  检测到: {len(turn_events)}/{len(all_runs)} runs')
    if no_turn:
        print(f'  失败:   {no_turn}')

    # [3] 派生
    print('\n[3/3] 派生任务数据集...')
    tasks = list(TASK_FUNCS.keys()) if args.task == 'all' else [args.task]
    all_summaries = {}

    for task_name in tasks:
        sm = process_task(task_name, all_runs, turn_events, args)
        all_summaries[task_name] = sm

    # 总结
    print('\n' + '=' * 70)
    print('  全部完成!')
    print('=' * 70)
    for tn, sm in all_summaries.items():
        total = sum(s['frames'] for s in sm['splits'].values())
        runs = sum(s['runs'] for s in sm['splits'].values())
        skipped = sm['skipped_count']
        print(f'  {tn + "_v1":25s}  {runs:3d} runs  {total:5d} 帧  '
              f'(跳过 {skipped})')

    print(f'\n  输出目录: {os.path.abspath(args.dst_root)}/')
    for tn in tasks:
        td = os.path.join(args.dst_root, f'{tn}_v1')
        print(f'    {tn}_v1/')
        for sp in SPLITS:
            spd = os.path.join(td, sp)
            if os.path.isdir(spd):
                n = len([d for d in os.listdir(spd) if os.path.isdir(
                    os.path.join(spd, d))])
                print(f'      {sp}/ ({n} runs)')
    print('=' * 70)


if __name__ == '__main__':
    main()
