"""
长直行纠偏回归数据集派生脚本
========================================
从新采集的长直行纠偏 runs 中，派生 straight_keep_reg_v1 回归数据集。

目标：训练一个回归 angular_z 的 SNN 模型，用于长时间直行保持不偏。

输入：
  data/straight_keep_raw/
    train/<run_name>/{images/, labels.csv, meta.json}
    val/<run_name>/...
    test/<run_name>/...

输出：
  data/straight_keep/straight_keep_reg_v1/
    train/<run_name>/{images/, labels.csv, meta.json}
    val/<run_name>/...
    test/<run_name>/...
    dataset_summary.json
    skipped_runs.csv

用法：
  # 派生全部
  python scripts/derive_straight_keep_dataset.py \\
      --src_root ./data/straight_keep_raw \\
      --dst_root ./data/straight_keep/straight_keep_reg_v1 \\
      --force

  # 自定义阶段检测参数
  python scripts/derive_straight_keep_dataset.py \\
      --src_root ./data/straight_keep_raw \\
      --dst_root ./data/straight_keep/straight_keep_reg_v1 \\
      --settle_window_ms 2000 --max_settled_frames 30 --force
"""

import os
import sys
import csv
import json
import math
import shutil
import random
import argparse
from collections import defaultdict


# ============================================================================
# 常量
# ============================================================================
SPLITS = ['train', 'val', 'test']


# ============================================================================
# 数据读取
# ============================================================================

def load_labels(csv_path, valid_only=True):
    """
    读取 labels.csv，返回帧列表。

    Returns:
        list of dict
    """
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
                'action_id': int(row.get('action_id', '0')),
                'action_name': row.get('action_name', ''),
                'timestamp_ns': int(row['timestamp_ns']),
                'linear_x': float(row.get('linear_x', '0')),
                'angular_z': float(row.get('angular_z', '0')),
                'time_diff_ms': float(row.get('time_diff_ms', '0')),
                'valid': valid,
            })
    return frames


def read_odom_csv(odom_path):
    """
    读取 odom_raw.csv，返回里程计记录列表。

    CSV 字段: timestamp_ns, x, y, yaw, linear_v, angular_w

    若文件不存在则返回 None（而非空列表），以区分"无 odom"和"odom 为空"。

    Returns:
        list of dict 或 None
    """
    if not os.path.isfile(odom_path):
        return None

    records = []
    try:
        with open(odom_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    'timestamp_ns': int(row['timestamp_ns']),
                    'x': float(row.get('x', '0')),
                    'y': float(row.get('y', '0')),
                    'yaw': float(row.get('yaw', '0')),
                    'linear_v': float(row.get('linear_v', '0')),
                    'angular_w': float(row.get('angular_w', '0')),
                })
    except Exception as e:
        print(f"  ⚠ 读取 odom_raw.csv 失败: {e}")
        return None

    return records if records else None


def compute_odom_stats(odom_records, t_start_ns=None, t_end_ns=None):
    """
    从 odom 记录中计算统计量（可选按时间窗口裁剪）。

    Args:
        odom_records: read_odom_csv 返回的列表
        t_start_ns: 起始时间戳 (纳秒)，None 则不裁剪
        t_end_ns:   结束时间戳 (纳秒)，None 则不裁剪

    Returns:
        dict: odom 统计结果
    """
    if not odom_records:
        return None

    # 按时间窗口裁剪
    recs = odom_records
    if t_start_ns is not None:
        recs = [r for r in recs if r['timestamp_ns'] >= t_start_ns]
    if t_end_ns is not None:
        recs = [r for r in recs if r['timestamp_ns'] <= t_end_ns]

    if len(recs) < 2:
        return None

    # 时间跨度
    dt_s = (recs[-1]['timestamp_ns'] - recs[0]['timestamp_ns']) / 1e9
    if dt_s <= 0:
        return None

    # 相对 yaw 变化 (累计绝对变化量)
    yaw_vals = [r['yaw'] for r in recs]
    yaw_abs_change = 0.0
    for i in range(1, len(yaw_vals)):
        dy = yaw_vals[i] - yaw_vals[i - 1]
        # 处理 yaw 环绕 (-π, π)
        while dy > math.pi:
            dy -= 2 * math.pi
        while dy < -math.pi:
            dy += 2 * math.pi
        yaw_abs_change += abs(dy)

    yaw_net_change = yaw_vals[-1] - yaw_vals[0]
    # 处理环绕
    while yaw_net_change > math.pi:
        yaw_net_change -= 2 * math.pi
    while yaw_net_change < -math.pi:
        yaw_net_change += 2 * math.pi

    # 路径长度
    path_length = 0.0
    for i in range(1, len(recs)):
        dx = recs[i]['x'] - recs[i - 1]['x']
        dy_pos = recs[i]['y'] - recs[i - 1]['y']
        path_length += math.sqrt(dx * dx + dy_pos * dy_pos)

    # 平均角速度 proxy
    angular_w_vals = [r['angular_w'] for r in recs]
    angular_w_mean = sum(angular_w_vals) / len(angular_w_vals)
    angular_w_abs_mean = sum(abs(w) for w in angular_w_vals) / len(angular_w_vals)

    # yaw 范围
    yaw_min = min(yaw_vals)
    yaw_max = max(yaw_vals)

    return {
        'odom_points': len(recs),
        'odom_duration_s': round(dt_s, 3),
        'yaw_net_change_rad': round(yaw_net_change, 6),
        'yaw_abs_change_rad': round(yaw_abs_change, 6),
        'yaw_range_rad': round(yaw_max - yaw_min, 6),
        'yaw_min_rad': round(yaw_min, 6),
        'yaw_max_rad': round(yaw_max, 6),
        'path_length_m': round(path_length, 4),
        'angular_w_mean': round(angular_w_mean, 6),
        'angular_w_abs_mean': round(angular_w_abs_mean, 6),
    }


def ns_to_ms(ns):
    """纳秒转毫秒"""
    return ns / 1e6


# ============================================================================
# 帧裁剪 & 阶段分类
# ============================================================================

def trim_buffer(frames, trim_start_ms, trim_end_ms):
    """
    去掉开头和结尾的缓冲段。

    策略:
      - 丢弃 t < t_first + trim_start_ms
      - 丢弃 t > t_last - trim_end_ms
    """
    if not frames:
        return []

    t0 = frames[0]['timestamp_ns']
    t1 = frames[-1]['timestamp_ns']
    start_cutoff = t0 + trim_start_ms * 1e6
    end_cutoff = t1 - trim_end_ms * 1e6

    if start_cutoff >= end_cutoff:
        # 裁剪后已经没有了
        return []

    return [f for f in frames if start_cutoff <= f['timestamp_ns'] <= end_cutoff]


def classify_correcting_settled(frames, settle_window_ms, w_settle_threshold=0.05):
    """
    将帧分为 Correcting 和 Settled 两个阶段。

    策略:
      从末尾向前扫描，找到最后一段连续 |angular_z| < threshold
      的窗口作为 Settled，其余为 Correcting。

    细节:
      1. 从最后一帧向前，找到第一个 |angular_z| >= threshold 的帧 idx_break
      2. idx_break+1 到末尾即为 Settled 候选
      3. 若 Settled 候选的时间跨度 >= settle_window_ms，确认为 Settled
      4. 否则全部标为 Correcting

    Returns:
        list of str: 每帧的阶段标签 ('Correcting' 或 'Settled')
    """
    n = len(frames)
    if n == 0:
        return []

    phases = ['Correcting'] * n

    # 从末尾向前扫描
    idx_break = n - 1
    while idx_break >= 0:
        if abs(frames[idx_break]['angular_z']) >= w_settle_threshold:
            break
        idx_break -= 1

    # idx_break+1 ... n-1 都是 |w| < threshold
    settled_start = idx_break + 1
    if settled_start < n:
        # 检查时间跨度
        t_settled_begin = ns_to_ms(frames[settled_start]['timestamp_ns'])
        t_settled_end = ns_to_ms(frames[-1]['timestamp_ns'])
        duration_ms = t_settled_end - t_settled_begin

        if duration_ms >= settle_window_ms:
            for i in range(settled_start, n):
                phases[i] = 'Settled'

    return phases


def cap_settled_frames(frames, phases, max_settled_frames):
    """
    限制 Settled 阶段的最大帧数。

    保留最后 max_settled_frames 帧的 Settled，
    多余的前面 Settled 帧丢弃。

    Returns:
        (new_frames, new_phases): 裁剪后的帧和阶段
    """
    # 找到所有 Settled 帧的索引
    settled_indices = [i for i, p in enumerate(phases) if p == 'Settled']

    if len(settled_indices) <= max_settled_frames:
        return frames, phases

    # 保留最后 max_settled_frames 个 Settled 帧
    keep_settled = set(settled_indices[-max_settled_frames:])
    drop_settled = set(settled_indices[:-max_settled_frames])

    new_frames = []
    new_phases = []
    for i in range(len(frames)):
        if i in drop_settled:
            continue
        new_frames.append(frames[i])
        new_phases.append(phases[i])

    return new_frames, new_phases


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


def write_derived_run(out_dir, frames_out, src_img_dir, copy_mode,
                      odom_stats=None):
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
        'image_name', 'timestamp_ns',
        'linear_x', 'angular_z',
        'action_id', 'action_name',
        'time_diff_ms', 'valid',
        'phase', 't_rel_ms', 'run_name', 'split',
    ]
    with open(os.path.join(out_dir, 'labels.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fr in frames_out:
            w.writerow({k: fr.get(k, '') for k in fieldnames})

    # angular_z 统计
    az_vals = [fr['angular_z'] for fr in frames_out]
    n = len(az_vals)
    az_mean = sum(az_vals) / max(n, 1)
    az_std = math.sqrt(sum((v - az_mean) ** 2 for v in az_vals) / max(n, 1))
    az_abs_mean = sum(abs(v) for v in az_vals) / max(n, 1)

    phase_dist = defaultdict(int)
    for fr in frames_out:
        phase_dist[fr['phase']] += 1

    # meta.json
    meta = {
        'total_frames': n,
        'valid_frames': n,
        'angular_z_stats': {
            'mean': round(az_mean, 6),
            'std': round(az_std, 6),
            'abs_mean': round(az_abs_mean, 6),
            'min': round(min(az_vals), 6) if az_vals else 0,
            'max': round(max(az_vals), 6) if az_vals else 0,
        },
        'phase_distribution': dict(phase_dist),
    }

    # odom 统计（若 odom_stats 通过 kwargs 传入）
    if odom_stats is not None:
        meta['odom_available'] = True
        meta['odom_stats'] = odom_stats
    else:
        meta['odom_available'] = False

    with open(os.path.join(out_dir, 'meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return n


# ============================================================================
# 单 run 处理
# ============================================================================

def process_run(run_name, run_dir, split, args):
    """
    处理单个 run 的派生。

    Returns:
        (frames_out, info_dict) 或 (None, reason_str)
        info_dict 中若存在 odom，会增加 'odom_stats' 字段
    """
    # 读取
    csv_path = os.path.join(run_dir, 'labels.csv')
    frames = load_labels(csv_path, valid_only=args.valid_only)

    if not frames:
        return None, '无有效帧'

    if len(frames) < 3:
        return None, f'帧数过少 ({len(frames)})'

    # 按时间排序
    frames.sort(key=lambda x: x['timestamp_ns'])

    original_count = len(frames)

    # 去掉开头/结尾缓冲
    frames = trim_buffer(frames, args.trim_start_ms, args.trim_end_ms)
    if not frames:
        return None, '缓冲裁剪后无帧'

    trimmed_count = len(frames)

    # 阶段检测: Correcting vs Settled
    # 动态阈值: 用本 run 的 angular_z 标准差来确定
    az_vals = [abs(f['angular_z']) for f in frames]
    az_median = sorted(az_vals)[len(az_vals) // 2]
    # 使用 median 的 30% 作为 settle 阈值, 最低 0.02, 最高 0.15
    w_settle_threshold = max(0.02, min(0.15, az_median * 0.3))

    phases = classify_correcting_settled(
        frames, args.settle_window_ms, w_settle_threshold)

    # 限制 Settled 帧数
    frames, phases = cap_settled_frames(
        frames, phases, args.max_settled_frames)

    if not frames:
        return None, '阶段裁剪后无帧'

    # 构建输出帧
    t0_ns = frames[0]['timestamp_ns']
    frames_out = []
    for i, fr in enumerate(frames):
        t_rel = ns_to_ms(fr['timestamp_ns']) - ns_to_ms(t0_ns)
        out = dict(fr)  # 保留原始字段
        out['phase'] = phases[i]
        out['t_rel_ms'] = round(t_rel, 1)
        out['run_name'] = run_name
        out['split'] = split
        frames_out.append(out)

    # 统计
    n_correcting = sum(1 for p in phases if p == 'Correcting')
    n_settled = sum(1 for p in phases if p == 'Settled')
    az_out = [f['angular_z'] for f in frames_out]
    az_mean = sum(az_out) / max(len(az_out), 1)
    az_abs_mean = sum(abs(v) for v in az_out) / max(len(az_out), 1)

    info = {
        'original_frames': original_count,
        'trimmed_frames': trimmed_count,
        'output_frames': len(frames_out),
        'correcting': n_correcting,
        'settled': n_settled,
        'w_settle_threshold': round(w_settle_threshold, 4),
        'az_mean': round(az_mean, 4),
        'az_abs_mean': round(az_abs_mean, 4),
    }

    # ---- odom 附加统计 (可选，不影响 phase 判定) ----
    odom_path = os.path.join(run_dir, 'odom_raw.csv')
    odom_records = read_odom_csv(odom_path)
    if odom_records is not None:
        # 用输出帧的时间窗口裁剪 odom
        t_start = frames_out[0]['timestamp_ns']
        t_end = frames_out[-1]['timestamp_ns']
        odom_st = compute_odom_stats(odom_records, t_start, t_end)
        if odom_st is not None:
            info['odom_stats'] = odom_st

    return frames_out, info


# ============================================================================
# 可调用入口 (供 pipeline 调用)
# ============================================================================

def run_derive_straight_keep(src_root, dst_root,
                              valid_only=True,
                              trim_start_ms=500,
                              trim_end_ms=500,
                              settle_window_ms=1500,
                              max_settled_frames=20,
                              copy_mode='copy',
                              seed=42,
                              force=False):
    """
    直行纠偏回归数据集派生核心函数。

    Returns:
        dict: dataset_summary
    """
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"源目录不存在: {src_root}")

    random.seed(seed)

    # 构造参数对象 (兼容 process_run)
    class Args:
        pass
    args_obj = Args()
    args_obj.valid_only = valid_only
    args_obj.trim_start_ms = trim_start_ms
    args_obj.trim_end_ms = trim_end_ms
    args_obj.settle_window_ms = settle_window_ms
    args_obj.max_settled_frames = max_settled_frames

    # ======================== Banner ========================
    print('=' * 72)
    print('  长直行纠偏回归数据集派生')
    print('=' * 72)
    print(f'  输入:             {os.path.abspath(src_root)}')
    print(f'  输出:             {os.path.abspath(dst_root)}')
    print(f'  缓冲裁剪:         start={trim_start_ms}ms, end={trim_end_ms}ms')
    print(f'  Settle 检测:      window={settle_window_ms}ms')
    print(f'  Settled 帧上限:   {max_settled_frames}')
    print(f'  复制方式:         {copy_mode}')
    print(f'  种子:             {seed}')
    print('=' * 72)

    # ======================== 扫描 ========================
    print('\n[1/3] 扫描 runs...')

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
            if not os.path.isfile(os.path.join(rd, 'labels.csv')):
                continue
            all_runs.append({
                'run_name': rn,
                'run_dir': rd,
                'split': split,
            })

    if not all_runs:
        raise RuntimeError("未找到任何有效 run!")

    split_counts = defaultdict(int)
    for r in all_runs:
        split_counts[r['split']] += 1
    for sp, cnt in sorted(split_counts.items()):
        print(f'  {sp}: {cnt} runs')
    print(f'  总计: {len(all_runs)} runs')

    # ======================== 派生 ========================
    print('\n[2/3] 派生 straight_keep_reg_v1...')

    split_stats = {s: {
        'runs': 0, 'frames': 0,
        'correcting': 0, 'settled': 0,
        'az_sum': 0.0, 'az_sq_sum': 0.0, 'az_abs_sum': 0.0,
    } for s in SPLITS}

    run_details = []
    skipped = []

    for r in all_runs:
        rn = r['run_name']
        sp = r['split']

        result, info = process_run(rn, r['run_dir'], sp, args_obj)

        if result is None or len(result) == 0:
            reason = info if isinstance(info, str) else '派生结果为空'
            skipped.append({'run_name': rn, 'split': sp, 'reason': reason})
            print(f'  [跳过] {rn} ({sp}): {reason}')
            continue

        # 写出
        out_run_dir = os.path.join(dst_root, sp, rn)
        if force and os.path.exists(out_run_dir):
            shutil.rmtree(out_run_dir)

        src_img_dir = os.path.join(r['run_dir'], 'images')
        run_odom_stats = info.get('odom_stats', None) if isinstance(info, dict) else None
        n_out = write_derived_run(out_run_dir, result, src_img_dir, copy_mode,
                                 odom_stats=run_odom_stats)

        # 统计
        st = split_stats[sp]
        st['runs'] += 1
        st['frames'] += n_out
        st['correcting'] += info['correcting']
        st['settled'] += info['settled']
        for fr in result:
            az = fr['angular_z']
            st['az_sum'] += az
            st['az_sq_sum'] += az * az
            st['az_abs_sum'] += abs(az)

        run_details.append({
            'run_name': rn,
            'split': sp,
            **info,
        })

        corr_pct = 100 * info['correcting'] / max(info['output_frames'], 1)
        odom_tag = ''
        if 'odom_stats' in info:
            os_ = info['odom_stats']
            odom_tag = (f'  odom: Δyaw={os_["yaw_net_change_rad"]:+.3f}rad '
                        f'path={os_["path_length_m"]:.2f}m')
        print(f'  [{sp:5s}] {rn:30s}: {n_out:4d} 帧  '
              f'(Corr={info["correcting"]}/{corr_pct:.0f}% '
              f'Settl={info["settled"]} '
              f'|w|={info["az_abs_mean"]:.3f} '
              f'θ_s={info["w_settle_threshold"]:.3f})'
              f'{odom_tag}')

    # ======================== 保存元数据 ========================
    print('\n[3/3] 保存元数据...')
    os.makedirs(dst_root, exist_ok=True)

    # 汇总统计
    summary_splits = {}
    total_runs = 0
    total_frames = 0
    total_correcting = 0
    total_settled = 0

    for sp in SPLITS:
        st = split_stats[sp]
        if st['runs'] == 0:
            continue
        n = st['frames']
        az_mean = st['az_sum'] / max(n, 1)
        az_std = math.sqrt(st['az_sq_sum'] / max(n, 1) - az_mean ** 2) \
            if n > 1 else 0.0
        az_abs_mean = st['az_abs_sum'] / max(n, 1)

        summary_splits[sp] = {
            'runs': st['runs'],
            'frames': st['frames'],
            'correcting': st['correcting'],
            'settled': st['settled'],
            'correcting_ratio': round(st['correcting'] / max(n, 1), 4),
            'settled_ratio': round(st['settled'] / max(n, 1), 4),
            'angular_z_stats': {
                'mean': round(az_mean, 6),
                'std': round(max(az_std, 0), 6),
                'abs_mean': round(az_abs_mean, 6),
            },
        }
        total_runs += st['runs']
        total_frames += st['frames']
        total_correcting += st['correcting']
        total_settled += st['settled']

    summary = {
        'task': 'straight_keep_reg',
        'version': 'v1',
        'regression_target': 'angular_z',
        'params': {
            'trim_start_ms': trim_start_ms,
            'trim_end_ms': trim_end_ms,
            'settle_window_ms': settle_window_ms,
            'max_settled_frames': max_settled_frames,
            'valid_only': valid_only,
            'seed': seed,
        },
        'splits': summary_splits,
        'total': {
            'runs': total_runs,
            'frames': total_frames,
            'correcting': total_correcting,
            'settled': total_settled,
        },
        'per_run': run_details,
        'skipped_count': len(skipped),
        'skipped_runs': skipped,
    }

    with open(os.path.join(dst_root, 'dataset_summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'  [✓] dataset_summary.json')

    # skipped_runs.csv
    if skipped:
        spath = os.path.join(dst_root, 'skipped_runs.csv')
        with open(spath, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['run_name', 'split', 'reason'])
            w.writeheader()
            w.writerows(skipped)
        print(f'  [✓] skipped_runs.csv ({len(skipped)} runs)')

    # ======================== odom 汇总 ========================
    # 统计多少 run 有 odom 信息
    odom_run_count = sum(1 for d in run_details if 'odom_stats' in d)
    odom_summary = None
    if odom_run_count > 0:
        # 汇总所有有 odom 的 run 的统计
        yaw_nets = [d['odom_stats']['yaw_net_change_rad']
                    for d in run_details if 'odom_stats' in d]
        yaw_abs_changes = [d['odom_stats']['yaw_abs_change_rad']
                           for d in run_details if 'odom_stats' in d]
        path_lengths = [d['odom_stats']['path_length_m']
                        for d in run_details if 'odom_stats' in d]
        angular_w_abs_means = [d['odom_stats']['angular_w_abs_mean']
                               for d in run_details if 'odom_stats' in d]

        odom_summary = {
            'odom_available_runs': odom_run_count,
            'odom_total_runs': len(run_details),
            'yaw_net_change_rad': {
                'mean': round(sum(yaw_nets) / len(yaw_nets), 6),
                'min': round(min(yaw_nets), 6),
                'max': round(max(yaw_nets), 6),
            },
            'yaw_abs_change_rad_mean': round(
                sum(yaw_abs_changes) / len(yaw_abs_changes), 6),
            'path_length_m': {
                'mean': round(sum(path_lengths) / len(path_lengths), 4),
                'min': round(min(path_lengths), 4),
                'max': round(max(path_lengths), 4),
            },
            'angular_w_abs_mean_avg': round(
                sum(angular_w_abs_means) / len(angular_w_abs_means), 6),
        }
        summary['odom_summary'] = odom_summary

    # ======================== 总结 ========================
    print(f'\n{"=" * 72}')
    print(f'  派生完成!')
    print(f'{"=" * 72}')

    for sp in SPLITS:
        if sp not in summary_splits:
            continue
        ss = summary_splits[sp]
        corr_pct = 100 * ss['correcting_ratio']
        settl_pct = 100 * ss['settled_ratio']
        az = ss['angular_z_stats']
        print(f'  {sp:5s}: {ss["runs"]:3d} runs  {ss["frames"]:5d} 帧  '
              f'Corr={corr_pct:.0f}% Settl={settl_pct:.0f}%  '
              f'w̄={az["mean"]:+.4f} σ={az["std"]:.4f} |w̄|={az["abs_mean"]:.4f}')

    print(f'  {"─" * 60}')
    print(f'  Total:  {total_runs:3d} runs  {total_frames:5d} 帧')
    if skipped:
        print(f'  跳过:   {len(skipped)} runs')

    # odom 汇总打印
    if odom_summary:
        print(f'\n  ── Odom 统计 ({odom_run_count}/{len(run_details)} runs 含 odom) ──')
        yn = odom_summary['yaw_net_change_rad']
        pl = odom_summary['path_length_m']
        print(f'    yaw 净变化:     '
              f'mean={yn["mean"]:+.4f}rad  '
              f'range=[{yn["min"]:+.4f}, {yn["max"]:+.4f}]')
        print(f'    yaw 累计变化:   '
              f'mean={odom_summary["yaw_abs_change_rad_mean"]:.4f}rad')
        print(f'    路径长度:       '
              f'mean={pl["mean"]:.3f}m  '
              f'range=[{pl["min"]:.3f}, {pl["max"]:.3f}]')
        print(f'    平均|angular_w|: '
              f'{odom_summary["angular_w_abs_mean_avg"]:.4f} rad/s')
    elif len(run_details) > 0:
        print(f'\n  ── Odom: 无 (所有 run 均未找到 odom_raw.csv) ──')

    print(f'\n  输出目录: {os.path.abspath(dst_root)}/')
    for sp in SPLITS:
        spd = os.path.join(dst_root, sp)
        if os.path.isdir(spd):
            n = len([d for d in os.listdir(spd)
                     if os.path.isdir(os.path.join(spd, d))])
            print(f'    {sp}/ ({n} runs)')

    print(f'\n  下一步训练命令:')
    print(f'    python train.py --dataset corridor \\\n'
          f'        --corridor_root {dst_root} \\\n'
          f'        --mode regression --control_dim 1 \\\n'
          f'        --neuron_type APLIF --residual_mode ADD \\\n'
          f'        -T 8 --encoding rate --loss_type huber \\\n'
          f'        --img_h 32 --img_w 96 \\\n'
          f'        -b 32 -epochs 100 \\\n'
          f'        -enable_tensorboard --final_test')
    print(f'{"=" * 72}')

    return summary


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='长直行纠偏回归数据集派生 '
                    '(输入: 已划分的 straight_keep 原始数据)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_root', type=str,
                        default='./data/straight_keep_raw',
                        help='已划分的直行数据根目录 (含 train/val/test/)')
    parser.add_argument('--dst_root', type=str,
                        default='./data/straight_keep/straight_keep_reg_v1',
                        help='派生输出根目录')
    parser.add_argument('--valid_only', type=lambda x:
                        str(x).lower() in ('true', '1', 'yes'),
                        default=True, help='丢弃 valid=0 帧')

    # 缓冲裁剪
    parser.add_argument('--trim_start_ms', type=float, default=500,
                        help='去掉开头缓冲段 (ms)')
    parser.add_argument('--trim_end_ms', type=float, default=500,
                        help='去掉结尾缓冲段 (ms)')

    # 阶段检测
    parser.add_argument('--settle_window_ms', type=float, default=1500,
                        help='末尾 Settled 阶段的最小时间窗口 (ms)')
    parser.add_argument('--max_settled_frames', type=int, default=20,
                        help='Settled 阶段最多保留帧数')

    # 输出
    parser.add_argument('--copy_mode', type=str, default='copy',
                        choices=['copy', 'symlink'],
                        help='图片复制方式 (推荐 copy)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true',
                        help='覆盖已有输出目录')

    args = parser.parse_args()

    run_derive_straight_keep(
        src_root=args.src_root,
        dst_root=args.dst_root,
        valid_only=args.valid_only,
        trim_start_ms=args.trim_start_ms,
        trim_end_ms=args.trim_end_ms,
        settle_window_ms=args.settle_window_ms,
        max_settled_frames=args.max_settled_frames,
        copy_mode=args.copy_mode,
        seed=args.seed,
        force=args.force,
    )


if __name__ == '__main__':
    main()
