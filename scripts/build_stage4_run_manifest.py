"""
Stage4 Run Manifest 生成器
========================================
为 stage4 四阶段数据集构建 run 级 manifest，
包含 turn 事件检测、阶段帧数统计和 delay 分桶信息。

支持 run 名格式:
  J1_left_r03  (标准 junction 格式)
  left1_bag3   (旧版兼容格式)

支持目录结构:
  flat:  src_root/<run_name>/
  split: src_root/{train,val,test}/<run_name>/

用法:
  python scripts/build_stage4_run_manifest.py \\
      --src_root ./data/corridor/

  python scripts/build_stage4_run_manifest.py \\
      --src_root ./data/junction_all \\
      --out_csv ./data/stage4_run_manifest.csv \\
      --pre_turn_ms 2000 --recover_ms 1800 \\
      --allow_unknown
"""

import os
import re
import csv
import json
import argparse
from collections import defaultdict


# ============================================================================
# 常量 & 正则
# ============================================================================

SPLITS = ['train', 'val', 'test']

# 标准格式: J1_left_r03
_RE_JUNCTION = re.compile(
    r'^J(?P<jid>\d+)_(?P<turn>left|right)_r(?P<rep>\d+)$',
    re.IGNORECASE)

# 旧版格式: left1_bag3
_RE_LEGACY = re.compile(
    r'^(?P<turn>left|right)(?P<jid>\d+)_bag(?P<rep>\d+)$',
    re.IGNORECASE)

# delay 分桶边界 (ms)
DELAY_BUCKETS = [
    (0,     500,   'very_short'),
    (500,   1500,  'short'),
    (1500,  3000,  'medium'),
    (3000,  5000,  'long'),
    (5000,  float('inf'), 'very_long'),
]


def _classify_delay(pre_ms):
    """按 pre_turn_ms_available 分桶"""
    for lo, hi, label in DELAY_BUCKETS:
        if lo <= pre_ms < hi:
            return label
    return 'unknown'


# ============================================================================
# Run 名解析
# ============================================================================

def parse_junction_run_name(name):
    """
    解析 junction run 目录名。

    支持:
      J1_left_r03  -> junction_id=1, turn_dir=left, rep_id=3
      left1_bag3   -> junction_id=1, turn_dir=left, rep_id=3

    Returns:
        dict 或 None
    """
    m = _RE_JUNCTION.match(name)
    if m:
        return {
            'junction_id': int(m.group('jid')),
            'turn_dir': m.group('turn').lower(),
            'rep_id': int(m.group('rep')),
        }
    m = _RE_LEGACY.match(name)
    if m:
        return {
            'junction_id': int(m.group('jid')),
            'turn_dir': m.group('turn').lower(),
            'rep_id': int(m.group('rep')),
        }
    return None


# ============================================================================
# 数据读取 (复用 derive_stage1 的格式)
# ============================================================================

def load_run_labels(run_dir, valid_only=True):
    """
    读取 run_dir/labels.csv 的帧列表。

    Returns:
        list of dict
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
                'action_id': int(row.get('action_id', '-1')),
                'action_name': row.get('action_name', ''),
                'timestamp_ns': int(row['timestamp_ns']),
                'linear_x': float(row.get('linear_x', 0)),
                'angular_z': float(row.get('angular_z', 0)),
                'time_diff_ms': float(row.get('time_diff_ms', 0)),
                'valid': valid,
            })
    return frames


def load_run_meta(run_dir):
    """读取 meta.json (若存在)"""
    path = os.path.join(run_dir, 'meta.json')
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


# ============================================================================
# Turn 检测 (复用 derive_stage1 逻辑)
# ============================================================================

def detect_turn_event(frames, k=3, w_threshold=0.3):
    """
    检测单个 run 的 turn 事件。

    策略 (按优先级):
      1. action_name 连续 K 帧为 Left 或 Right
      2. |angular_z| >= threshold 且符号一致连续 K 帧

    Returns:
        dict 或 None
    """
    n = len(frames)
    if n < k:
        return None

    # ---- 方法1: action_name ----
    for i in range(n - k + 1):
        names = [frames[j]['action_name'] for j in range(i, i + k)]
        if all(nm in ('Left', 'Right') for nm in names) and \
                len(set(names)) == 1:
            direction = names[0].lower()
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

    # ---- 方法2: angular_z ----
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
                    if abs(nv) >= w_threshold and \
                            (nv > 0) == (vals[0] > 0):
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


# ============================================================================
# 阶段帧数统计
# ============================================================================

def count_phase_frames(frames, turn_event, pre_turn_ms=2000,
                       recover_ms=1800):
    """
    统计每个阶段的帧数。

    返回 Follow / Approach / Turn / Recover / Post 帧数。
    """
    if turn_event is None:
        return {
            'follow_frames_raw': len(frames),
            'approach_frames_raw': 0,
            'turn_frames_raw': 0,
            'recover_frames_raw': 0,
            'post_frames_raw': 0,
        }

    t_on = turn_event['t_turn_on_ns']
    t_off = turn_event['t_turn_off_ns']

    counts = defaultdict(int)
    for f in frames:
        t = f['timestamp_ns']
        t_ms = t / 1e6
        on_ms = t_on / 1e6
        off_ms = t_off / 1e6

        if t_ms < on_ms - pre_turn_ms:
            counts['follow'] += 1
        elif t_ms < on_ms:
            counts['approach'] += 1
        elif t_ms <= off_ms:
            counts['turn'] += 1
        elif t_ms <= off_ms + recover_ms:
            counts['recover'] += 1
        else:
            counts['post'] += 1

    return {
        'follow_frames_raw': counts['follow'],
        'approach_frames_raw': counts['approach'],
        'turn_frames_raw': counts['turn'],
        'recover_frames_raw': counts['recover'],
        'post_frames_raw': counts['post'],
    }


# ============================================================================
# 单 run 处理
# ============================================================================

def process_run(run_name, run_dir, pre_turn_ms=2000, recover_ms=1800,
                k=3, w_threshold=0.3):
    """
    处理单个 run，生成 manifest 记录。

    Returns:
        dict 或 None
    """
    # 读取帧
    all_frames = load_run_labels(run_dir, valid_only=False)
    valid_frames = load_run_labels(run_dir, valid_only=True)

    if not valid_frames:
        return None

    # 解析 run 名
    parsed = parse_junction_run_name(run_name)
    jid = parsed['junction_id'] if parsed else -1
    turn_dir_from_name = parsed['turn_dir'] if parsed else 'unknown'
    rep = parsed['rep_id'] if parsed else -1

    # Turn 检测
    te = detect_turn_event(valid_frames, k=k, w_threshold=w_threshold)

    # turn_dir: 优先用检测结果
    turn_dir = te['turn_dir'] if te else turn_dir_from_name

    # 时间信息
    t_turn_on = te['t_turn_on_ns'] if te else 0
    t_turn_off = te['t_turn_off_ns'] if te else 0
    turn_dur_ms = (t_turn_off - t_turn_on) / 1e6 if te else 0

    # 实际可用的 pre_turn 时长
    t_first = valid_frames[0]['timestamp_ns']
    pre_avail_ms = (t_turn_on - t_first) / 1e6 if te else 0

    # 阶段帧数
    phase_counts = count_phase_frames(
        valid_frames, te,
        pre_turn_ms=pre_turn_ms, recover_ms=recover_ms)

    has_follow = phase_counts['follow_frames_raw'] > 0
    has_all_4 = all(phase_counts[k] > 0 for k in [
        'follow_frames_raw', 'approach_frames_raw',
        'turn_frames_raw', 'recover_frames_raw'])

    delay_bucket = _classify_delay(pre_avail_ms) if te else 'no_turn'

    record = {
        'run_name': run_name,
        'junction_id': jid,
        'turn_dir': turn_dir,
        'rep_id': rep,
        'total_frames': len(all_frames),
        'valid_frames': len(valid_frames),
        't_turn_on_ns': t_turn_on,
        't_turn_off_ns': t_turn_off,
        'turn_duration_ms': round(turn_dur_ms, 1),
        'pre_turn_ms_available': round(pre_avail_ms, 1),
        'follow_frames_raw': phase_counts['follow_frames_raw'],
        'approach_frames_raw': phase_counts['approach_frames_raw'],
        'turn_frames_raw': phase_counts['turn_frames_raw'],
        'recover_frames_raw': phase_counts['recover_frames_raw'],
        'has_follow': has_follow,
        'has_all_4_phases': has_all_4,
        'delay_bucket': delay_bucket,
        'turn_detect_method': te['method'] if te else 'none',
    }

    return record


# ============================================================================
# 扫描 & 汇总
# ============================================================================

def scan_runs(src_root, allow_unknown=False, **kwargs):
    """
    扫描 src_root 下所有 run 目录并生成 manifest 记录。

    支持:
      flat:  src_root/<run_name>/
      split: src_root/{train,val,test}/<run_name>/
    """
    records = []
    skipped = []

    def _scan_dir(parent, found_in=''):
        if not os.path.isdir(parent):
            return
        for name in sorted(os.listdir(parent)):
            full = os.path.join(parent, name)
            if not os.path.isdir(full):
                continue

            # split 子目录
            if name in SPLITS:
                _scan_dir(full, found_in=name)
                continue

            # 必须有 labels.csv
            if not os.path.isfile(os.path.join(full, 'labels.csv')):
                continue

            # 尝试解析 run 名
            parsed = parse_junction_run_name(name)
            if parsed is None and not allow_unknown:
                skipped.append({
                    'run_name': name,
                    'reason': '格式不匹配',
                    'parent': found_in or '(root)',
                })
                continue

            record = process_run(name, full, **kwargs)
            if record:
                if found_in:
                    record['found_in'] = found_in
                records.append(record)
            else:
                skipped.append({
                    'run_name': name,
                    'reason': '无有效帧',
                    'parent': found_in or '(root)',
                })

    _scan_dir(src_root)
    return records, skipped


# ============================================================================
# 输出
# ============================================================================

CSV_FIELDS = [
    'run_name', 'junction_id', 'turn_dir', 'rep_id',
    'total_frames', 'valid_frames',
    't_turn_on_ns', 't_turn_off_ns', 'turn_duration_ms',
    'pre_turn_ms_available',
    'follow_frames_raw', 'approach_frames_raw',
    'turn_frames_raw', 'recover_frames_raw',
    'has_follow', 'has_all_4_phases', 'delay_bucket',
    'turn_detect_method',
]


def write_csv(out_csv, records):
    """写出 manifest CSV"""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def write_json(out_json, records, summary):
    """写出 manifest JSON"""
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    data = {
        'summary': summary,
        'runs': records,
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_summary(records, skipped):
    """打印统计摘要"""
    print(f'\n{"=" * 60}')
    print(f'  Stage4 Run Manifest 统计')
    print(f'{"=" * 60}')
    print(f'  总 run 数:     {len(records)}')

    if not records:
        print(f'  (无记录)')
        return {}

    # junction 分布
    jid_dist = defaultdict(int)
    for r in records:
        jid_dist[r['junction_id']] += 1
    print(f'\n  ── Junction 分布 ──')
    for jid in sorted(jid_dist.keys()):
        label = f'J{jid}' if jid >= 0 else '(unknown)'
        print(f'    {label:10s}: {jid_dist[jid]} runs')

    # turn_dir 分布
    dir_dist = defaultdict(int)
    for r in records:
        dir_dist[r['turn_dir']] += 1
    print(f'\n  ── Turn Direction 分布 ──')
    for td, cnt in sorted(dir_dist.items()):
        print(f'    {td:10s}: {cnt} runs')

    # has_all_4_phases
    n_all4 = sum(1 for r in records if r['has_all_4_phases'])
    n_no_turn = sum(1 for r in records
                    if r['turn_detect_method'] == 'none')
    print(f'\n  ── 阶段覆盖 ──')
    print(f'    四阶段完整:  {n_all4} / {len(records)}')
    print(f'    无 turn 事件: {n_no_turn}')

    # delay_bucket 分布
    bucket_dist = defaultdict(int)
    for r in records:
        bucket_dist[r['delay_bucket']] += 1
    print(f'\n  ── Delay 分桶 ──')
    for b in ['very_short', 'short', 'medium', 'long',
              'very_long', 'no_turn']:
        if b in bucket_dist:
            print(f'    {b:12s}: {bucket_dist[b]}')

    # 帧数统计
    follow_total = sum(r['follow_frames_raw'] for r in records)
    approach_total = sum(r['approach_frames_raw'] for r in records)
    turn_total = sum(r['turn_frames_raw'] for r in records)
    recover_total = sum(r['recover_frames_raw'] for r in records)
    all_total = follow_total + approach_total + turn_total + recover_total
    print(f'\n  ── 阶段帧数汇总 ──')
    if all_total > 0:
        for name, cnt in [('Follow', follow_total),
                          ('Approach', approach_total),
                          ('Turn', turn_total),
                          ('Recover', recover_total)]:
            pct = cnt / all_total * 100
            print(f'    {name:10s}: {cnt:7d}  ({pct:5.1f}%)')
        print(f'    {"Total":10s}: {all_total:7d}')

    # 跳过
    if skipped:
        print(f'\n  ── 跳过 ({len(skipped)}) ──')
        for s in skipped[:10]:
            print(f'    - {s["run_name"]}: {s["reason"]} '
                  f'({s["parent"]})')
        if len(skipped) > 10:
            print(f'    ... 还有 {len(skipped) - 10} 个')

    print(f'{"=" * 60}')

    summary = {
        'total_runs': len(records),
        'runs_with_all_4_phases': n_all4,
        'runs_without_turn': n_no_turn,
        'junction_distribution': dict(jid_dist),
        'turn_dir_distribution': dict(dir_dist),
        'delay_bucket_distribution': dict(bucket_dist),
        'phase_frame_totals': {
            'follow': follow_total,
            'approach': approach_total,
            'turn': turn_total,
            'recover': recover_total,
        },
    }
    return summary


# ============================================================================
# 主入口
# ============================================================================

def build_stage4_manifest(src_root, out_csv, out_json=None,
                          pre_turn_ms=2000, recover_ms=1800,
                          k=3, w_threshold=0.3,
                          allow_unknown=False):
    """
    构建 stage4 run manifest 核心函数。

    Returns:
        (records, skipped, summary)
    """
    print(f'{"=" * 60}')
    print(f'  Stage4 Run Manifest 生成器')
    print(f'{"=" * 60}')
    print(f'  数据目录:       {os.path.abspath(src_root)}')
    print(f'  输出 CSV:       {os.path.abspath(out_csv)}')
    if out_json:
        print(f'  输出 JSON:      {os.path.abspath(out_json)}')
    print(f'  pre_turn_ms:    {pre_turn_ms}')
    print(f'  recover_ms:     {recover_ms}')
    print(f'  turn 检测:      k={k}, w_threshold={w_threshold}')
    print(f'  allow_unknown:  {allow_unknown}')
    print(f'{"=" * 60}')

    print(f'\n[1/3] 扫描 run 目录...')
    records, skipped = scan_runs(
        src_root, allow_unknown=allow_unknown,
        pre_turn_ms=pre_turn_ms, recover_ms=recover_ms,
        k=k, w_threshold=w_threshold)

    if not records:
        print(f'  ✗ 未找到任何有效 run!')
        return records, skipped, {}

    print(f'  找到 {len(records)} 个有效 run')
    if skipped:
        print(f'  跳过 {len(skipped)} 个目录')

    print(f'\n[2/3] 写出 manifest...')
    write_csv(out_csv, records)
    print(f'  [✓] {out_csv}')

    if out_json:
        summary = print_summary(records, skipped)
        write_json(out_json, records, summary)
        print(f'  [✓] {out_json}')
    else:
        summary = {}

    print(f'\n[3/3] 统计...')
    summary = print_summary(records, skipped)

    print(f'\n  下一步:')
    print(f'    python scripts/split_corridor_runs.py \\')
    print(f'        --src_root {src_root} \\')
    print(f'        --manifest {out_csv} \\')
    print(f'        --group_by junction_id,turn_dir \\')
    print(f'        --split_mode ratio --val_ratio 0.15 '
          f'--test_ratio 0.15')

    return records, skipped, summary


def main():
    parser = argparse.ArgumentParser(
        description='为 stage4 构建 run 级 manifest — '
                    '含 turn 检测和阶段帧数统计',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_root', type=str,
                        default='./data/corridor_all',
                        help='原始 corridor runs 根目录')
    parser.add_argument('--out_csv', type=str,
                        default='./data/stage4_run_manifest.csv',
                        help='输出 manifest CSV 路径')
    parser.add_argument('--out_json', type=str, default=None,
                        help='输出 manifest JSON 路径 (可选)')
    parser.add_argument('--pre_turn_ms', type=float, default=2000,
                        help='turn_on 前的 Follow/Approach 窗口 (ms)')
    parser.add_argument('--recover_ms', type=float, default=1800,
                        help='turn_off 后的 Recover 窗口 (ms)')
    parser.add_argument('--turn_k', type=int, default=3,
                        help='turn 检测: 连续帧数')
    parser.add_argument('--w_threshold', type=float, default=0.3,
                        help='turn 检测: angular_z 阈值')
    parser.add_argument('--allow_unknown', action='store_true',
                        help='遇到无法解析的目录名时跳过而不报错')

    args = parser.parse_args()

    build_stage4_manifest(
        src_root=args.src_root,
        out_csv=args.out_csv,
        out_json=args.out_json,
        pre_turn_ms=args.pre_turn_ms,
        recover_ms=args.recover_ms,
        k=args.turn_k,
        w_threshold=args.w_threshold,
        allow_unknown=args.allow_unknown,
    )


if __name__ == '__main__':
    main()
