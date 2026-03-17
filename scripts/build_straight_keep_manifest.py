"""
Straight-Keep Manifest 生成器
========================================
从 straight_keep 数据目录中的 run 名自动生成 runs_manifest.csv，
供 split_corridor_runs.py 和 corridor_dataset_pipeline.py 使用。

支持 run 名格式:
  S{segment}_P{station}_{offset}_{yaw}_r{rep}
  例: S1_P1_C_Y0_r01, S2_P2_L10_Y0_r03, S4_P1_C_YL5_r02

输出 CSV 字段:
  run_name, segment_id, station_id, offset_cm, yaw_deg, condition, rep_id

用法:
  # 默认扫描当前目录
  python scripts/build_straight_keep_manifest.py \\
      --src_root ./data/straight_keep_all \\
      --out_csv ./data/straight_keep_manifest.csv

  # 允许跳过无法解析的目录
  python scripts/build_straight_keep_manifest.py \\
      --src_root ./data/straight_keep_all \\
      --out_csv ./data/straight_keep_manifest.csv \\
      --allow_unknown
"""

import os
import re
import csv
import argparse
from collections import defaultdict


# ============================================================================
# 常量 & 解析规则
# ============================================================================

# run 名正则: S1_P1_C_Y0_r01
_RE_RUN = re.compile(
    r'^S(?P<segment>\d+)_P(?P<station>\d+)_'
    r'(?P<offset>C|L\d+|R\d+)_'
    r'(?P<yaw>Y0|YL\d+|YR\d+)_'
    r'r(?P<rep>\d+)$',
    re.IGNORECASE
)

# 偏移量映射: 标签 -> cm
_OFFSET_MAP = {
    'C':   0,
    'L10': -10,
    'R10':  10,
    'L15': -15,
    'R15':  15,
    'L20': -20,
    'R20':  20,
    'L5':  -5,
    'R5':   5,
}

# 航向偏差映射: 标签 -> deg
_YAW_MAP = {
    'Y0':   0,
    'YL5': -5,
    'YR5':  5,
    'YL10': -10,
    'YR10':  10,
}


def _parse_offset(tag):
    """解析偏移量标签 -> (offset_cm, 原始标签)"""
    tag_upper = tag.upper()
    if tag_upper in _OFFSET_MAP:
        return _OFFSET_MAP[tag_upper], tag_upper
    # 尝试动态解析: L/R + 数字
    m = re.match(r'^([LR])(\d+)$', tag_upper)
    if m:
        sign = -1 if m.group(1) == 'L' else 1
        return sign * int(m.group(2)), tag_upper
    return None, tag_upper


def _parse_yaw(tag):
    """解析航向标签 -> (yaw_deg, 原始标签)"""
    tag_upper = tag.upper()
    if tag_upper in _YAW_MAP:
        return _YAW_MAP[tag_upper], tag_upper
    # 尝试动态解析: YL/YR + 数字
    m = re.match(r'^Y([LR])(\d+)$', tag_upper)
    if m:
        sign = -1 if m.group(1) == 'L' else 1
        return sign * int(m.group(2)), tag_upper
    return None, tag_upper


def parse_run_name(name):
    """
    解析 straight_keep run 名。

    Args:
        name: run 目录名

    Returns:
        dict 或 None
    """
    m = _RE_RUN.match(name)
    if not m:
        return None

    segment_id = int(m.group('segment'))
    station_id = int(m.group('station'))
    offset_tag = m.group('offset')
    yaw_tag = m.group('yaw')
    rep_id = int(m.group('rep'))

    offset_cm, offset_label = _parse_offset(offset_tag)
    yaw_deg, yaw_label = _parse_yaw(yaw_tag)

    if offset_cm is None or yaw_deg is None:
        return None

    # condition = offset_label + '_' + yaw_label
    condition = f'{offset_label}_{yaw_label}'

    return {
        'run_name': name,
        'segment_id': segment_id,
        'station_id': station_id,
        'offset_cm': offset_cm,
        'yaw_deg': yaw_deg,
        'condition': condition,
        'rep_id': rep_id,
    }


# ============================================================================
# 扫描 & 生成
# ============================================================================

def scan_runs(src_root, allow_unknown=False):
    """
    扫描 src_root 下所有有效 run 目录并解析。

    支持两种目录结构:
      flat:  src_root/<run_name>/
      split: src_root/{train,val,test}/<run_name>/

    Returns:
        (records, skipped)
    """
    records = []
    skipped = []
    seen_names = set()

    def _scan_dir(parent, prefix=''):
        """扫描一层目录"""
        if not os.path.isdir(parent):
            return
        for name in sorted(os.listdir(parent)):
            full = os.path.join(parent, name)
            if not os.path.isdir(full):
                continue

            # 检查是否为 split 子目录
            if name in ('train', 'val', 'test'):
                _scan_dir(full, prefix=name)
                continue

            # 尝试解析 run 名
            parsed = parse_run_name(name)
            if parsed is None:
                if not allow_unknown:
                    print(f'  ✗ 无法解析 run 名: {name}')
                    print(f'    期望格式: S{{seg}}_P{{sta}}_{{offset}}_{{yaw}}_r{{rep}}')
                    print(f'    例: S1_P1_C_Y0_r01, S2_P2_L10_YR5_r03')
                    raise ValueError(
                        f'run 名 "{name}" 不符合 straight_keep 格式，'
                        f'使用 --allow_unknown 跳过')
                skipped.append({
                    'run_name': name,
                    'reason': '格式不匹配',
                    'parent': prefix or '(root)',
                })
                continue

            # 重复检查
            if name in seen_names:
                print(f'  ⚠ 重复 run 名: {name}')
            seen_names.add(name)

            if prefix:
                parsed['found_in'] = prefix
            records.append(parsed)

    _scan_dir(src_root)
    return records, skipped


def write_manifest(out_csv, records):
    """写出 manifest CSV"""
    fieldnames = [
        'run_name', 'segment_id', 'station_id',
        'offset_cm', 'yaw_deg', 'condition', 'rep_id',
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def print_summary(records, skipped):
    """打印统计摘要"""
    print(f'\n{"=" * 60}')
    print(f'  Manifest 统计')
    print(f'{"=" * 60}')
    print(f'  总 run 数:  {len(records)}')

    if not records:
        print(f'  (无记录)')
        return

    # 按 segment 统计
    seg_dist = defaultdict(int)
    for r in records:
        seg_dist[r['segment_id']] += 1
    print(f'\n  ── Segment 分布 ──')
    for seg in sorted(seg_dist.keys()):
        print(f'    S{seg}: {seg_dist[seg]} runs')

    # 按 station 统计
    sta_dist = defaultdict(int)
    for r in records:
        sta_dist[r['station_id']] += 1
    print(f'\n  ── Station 分布 ──')
    for sta in sorted(sta_dist.keys()):
        print(f'    P{sta}: {sta_dist[sta]} runs')

    # 按 condition 统计
    cond_dist = defaultdict(int)
    for r in records:
        cond_dist[r['condition']] += 1
    print(f'\n  ── Condition 分布 ──')
    for cond in sorted(cond_dist.keys()):
        print(f'    {cond:12s}: {cond_dist[cond]} runs')

    # 按 segment x condition 交叉表
    seg_cond = defaultdict(lambda: defaultdict(int))
    for r in records:
        seg_cond[r['segment_id']][r['condition']] += 1

    all_conds = sorted(cond_dist.keys())
    print(f'\n  ── Segment × Condition 交叉表 ──')
    header = f'    {"Seg":>6s}'
    for c in all_conds:
        header += f'  {c:>8s}'
    header += f'  {"Total":>6s}'
    print(header)
    print(f'    {"─" * (len(header) - 4)}')
    for seg in sorted(seg_dist.keys()):
        row = f'    {"S" + str(seg):>6s}'
        total = 0
        for c in all_conds:
            cnt = seg_cond[seg][c]
            row += f'  {cnt:>8d}'
            total += cnt
        row += f'  {total:>6d}'
        print(row)

    # 重复检查
    names = [r['run_name'] for r in records]
    dups = [n for n in set(names) if names.count(n) > 1]
    if dups:
        print(f'\n  ⚠ 发现 {len(dups)} 个重复 run_name:')
        for d in sorted(dups):
            print(f'    - {d}')
    else:
        print(f'\n  ✓ 无重复 run_name')

    if skipped:
        print(f'\n  ── 跳过 ({len(skipped)}) ──')
        for s in skipped:
            print(f'    - {s["run_name"]}: {s["reason"]} ({s["parent"]})')

    print(f'{"=" * 60}')


# ============================================================================
# 主入口
# ============================================================================

def build_manifest(src_root, out_csv, allow_unknown=False):
    """
    构建 straight_keep manifest 核心函数。

    Returns:
        (records, skipped)
    """
    print(f'{"=" * 60}')
    print(f'  Straight-Keep Manifest 生成器')
    print(f'{"=" * 60}')
    print(f'  数据目录:   {os.path.abspath(src_root)}')
    print(f'  输出文件:   {os.path.abspath(out_csv)}')
    print(f'  allow_unknown: {allow_unknown}')
    print(f'{"=" * 60}')

    print(f'\n[1/3] 扫描 run 目录...')
    records, skipped = scan_runs(src_root, allow_unknown=allow_unknown)

    if not records:
        print(f'  ✗ 未找到任何有效 run!')
        return records, skipped

    print(f'  找到 {len(records)} 个有效 run')
    if skipped:
        print(f'  跳过 {len(skipped)} 个无法解析的目录')

    print(f'\n[2/3] 写出 manifest...')
    write_manifest(out_csv, records)
    print(f'  [✓] {out_csv}')

    print(f'\n[3/3] 统计...')
    print_summary(records, skipped)

    print(f'\n  下一步:')
    print(f'    python scripts/split_corridor_runs.py \\')
    print(f'        --src_root {src_root} \\')
    print(f'        --manifest {out_csv} \\')
    print(f'        --group_by segment_id,condition \\')
    print(f'        --split_mode ratio --val_ratio 0.2 --test_ratio 0.2')

    return records, skipped


def main():
    parser = argparse.ArgumentParser(
        description='从 straight_keep 数据目录生成 runs_manifest.csv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_root', type=str,
                        default='./data/straight_keep_all',
                        help='straight_keep 数据根目录 '
                             '(含 run 子目录或 train/val/test/ 子目录)')
    parser.add_argument('--out_csv', type=str,
                        default='./data/straight_keep_manifest.csv',
                        help='输出 manifest CSV 路径')
    parser.add_argument('--allow_unknown', action='store_true',
                        help='遇到无法解析的目录名时跳过而不报错')

    args = parser.parse_args()

    build_manifest(
        src_root=args.src_root,
        out_csv=args.out_csv,
        allow_unknown=args.allow_unknown,
    )


if __name__ == '__main__':
    main()
