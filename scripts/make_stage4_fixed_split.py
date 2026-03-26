"""
Stage4 固定划分生成器
========================================
根据 stage4_run_manifest.csv 自动生成固定 train/val/test 划分。

按 (junction_id, turn_dir) 分组，每组 7 run 时分配 5+1+1。
选 run 策略:
  - 按 pre_turn_ms_available 排序
  - val 选中位附近、优先 has_all_4_phases=True
  - test 选与 val 不同 delay_bucket 的、优先 has_all_4_phases
  - val/test 优先 has_follow=True

用法:
    python scripts/make_stage4_fixed_split.py \\
        --manifest_csv ./data/stage4_run_manifest.csv \\
        --out_csv ./data/stage4_fixed_split.csv
"""

import os
import csv
import argparse
from collections import defaultdict


# ============================================================================
# 读取 manifest
# ============================================================================

def load_manifest(csv_path):
    """
    读取 stage4_run_manifest.csv。

    Returns:
        list of dict (每个 run 一条记录)
    """
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换类型
            rec = {
                'run_name': row['run_name'].strip(),
                'junction_id': int(row.get('junction_id', -1)),
                'turn_dir': row.get('turn_dir', 'unknown').strip().lower(),
                'pre_turn_ms_available': float(
                    row.get('pre_turn_ms_available', 0)),
                'delay_bucket': row.get('delay_bucket', 'unknown').strip(),
                'has_follow': _parse_bool(row.get('has_follow', 'False')),
                'has_all_4_phases': _parse_bool(
                    row.get('has_all_4_phases', 'False')),
            }
            records.append(rec)
    return records


def _parse_bool(val):
    """将 CSV 中的 True/False/1/0 转为 bool"""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ('true', '1', 'yes')


# ============================================================================
# 划分策略
# ============================================================================

def _sort_key_pre_turn(run):
    """按 pre_turn_ms_available 排序"""
    return run['pre_turn_ms_available']


def _pick_val_test(runs_sorted, n_val=1, n_test=1):
    """
    从按 pre_turn_ms_available 排序的列表中选 val/test。

    策略:
      1. 优先从 has_all_4_phases=True 的 run 中选
      2. val 选中位附近
      3. test 选与 val 不同 delay_bucket
      4. val/test 优先 has_follow=True

    Returns:
        (val_names, test_names, train_names)
    """
    n = len(runs_sorted)

    # 候选池: 优先 has_all_4_phases，其次 has_follow
    prefer = [r for r in runs_sorted
              if r['has_all_4_phases'] and r['has_follow']]
    if len(prefer) < n_val + n_test:
        # 放宽: has_all_4_phases 即可
        prefer = [r for r in runs_sorted if r['has_all_4_phases']]
    if len(prefer) < n_val + n_test:
        # 再放宽: has_follow 即可
        prefer = [r for r in runs_sorted if r['has_follow']]
    if len(prefer) < n_val + n_test:
        # 最终: 所有 run
        prefer = list(runs_sorted)

    # ---- 选 val: 中位附近 ----
    mid_idx = len(prefer) // 2
    val_run = prefer[mid_idx]

    # ---- 选 test: 不同 delay_bucket ----
    val_bucket = val_run['delay_bucket']
    remaining = [r for r in prefer if r['run_name'] != val_run['run_name']]

    # 优先选不同 bucket
    diff_bucket = [r for r in remaining
                   if r['delay_bucket'] != val_bucket]
    if diff_bucket:
        # 从不同 bucket 中选距离中位最远的 (增加多样性)
        test_run = diff_bucket[0] if diff_bucket[0] != val_run else \
            diff_bucket[-1]
    elif remaining:
        # 无不同 bucket，选离 val 最远的
        test_run = remaining[0] if remaining[0] != val_run else remaining[-1]
    else:
        # 该组只有 1-2 个 run，强制选
        test_run = runs_sorted[-1] if runs_sorted[-1] != val_run else \
            runs_sorted[0]

    val_names = {val_run['run_name']}
    test_names = {test_run['run_name']}

    # 避免重复
    if val_run['run_name'] == test_run['run_name'] and n > 1:
        # 选另一个
        for r in runs_sorted:
            if r['run_name'] not in val_names:
                test_names = {r['run_name']}
                break

    # 剩余为 train
    train_names = {r['run_name'] for r in runs_sorted
                   if r['run_name'] not in val_names and
                   r['run_name'] not in test_names}

    return val_names, test_names, train_names


def generate_fixed_split(records):
    """
    根据 manifest 记录生成固定划分。

    Returns:
        list of dict (每个 run 一条, 含 split 字段)
    """
    # 按 (junction_id, turn_dir) 分组
    groups = defaultdict(list)
    for rec in records:
        key = (rec['junction_id'], rec['turn_dir'])
        groups[key].append(rec)

    result = []
    stats = {'train': 0, 'val': 0, 'test': 0}

    print(f'\n{"=" * 65}')
    print(f'  Stage4 固定划分生成')
    print(f'{"=" * 65}')
    print(f'  总 run 数: {len(records)}')
    print(f'  分组数:    {len(groups)}')
    print(f'{"=" * 65}')

    for key in sorted(groups.keys()):
        grp = groups[key]
        jid, td = key
        grp_label = f'J{jid}_{td}'

        # 按 pre_turn_ms_available 排序
        grp_sorted = sorted(grp, key=_sort_key_pre_turn)
        n = len(grp_sorted)

        if n < 3:
            # 太少: 全部给 train，打印警告
            print(f'\n  ⚠ {grp_label}: 仅 {n} runs，全部分配为 train')
            for r in grp_sorted:
                result.append(_make_row(r, 'train'))
                stats['train'] += 1
            continue

        # 计算分配数目
        if n >= 7:
            n_train = n - 2
            n_val = 1
            n_test = 1
        elif n >= 5:
            n_train = n - 2
            n_val = 1
            n_test = 1
        else:  # 3-4
            n_train = n - 2
            n_val = 1
            n_test = 1

        val_names, test_names, train_names = _pick_val_test(
            grp_sorted, n_val=n_val, n_test=n_test)

        # 输出
        pre_range = (f'{grp_sorted[0]["pre_turn_ms_available"]:.0f}'
                     f'~{grp_sorted[-1]["pre_turn_ms_available"]:.0f}ms')
        print(f'\n  {grp_label:12s}: {n} runs  '
              f'(pre_turn: {pre_range})')
        print(f'    train ({len(train_names)}): '
              f'{", ".join(sorted(train_names))}')

        for vn in sorted(val_names):
            vr = next(r for r in grp_sorted if r['run_name'] == vn)
            print(f'    val   (1): {vn}  '
                  f'[bucket={vr["delay_bucket"]}, '
                  f'4ph={vr["has_all_4_phases"]}, '
                  f'follow={vr["has_follow"]}]')

        for tn in sorted(test_names):
            tr = next(r for r in grp_sorted if r['run_name'] == tn)
            print(f'    test  (1): {tn}  '
                  f'[bucket={tr["delay_bucket"]}, '
                  f'4ph={tr["has_all_4_phases"]}, '
                  f'follow={tr["has_follow"]}]')

        for r in grp_sorted:
            if r['run_name'] in val_names:
                result.append(_make_row(r, 'val'))
                stats['val'] += 1
            elif r['run_name'] in test_names:
                result.append(_make_row(r, 'test'))
                stats['test'] += 1
            else:
                result.append(_make_row(r, 'train'))
                stats['train'] += 1

    # 汇总
    print(f'\n{"─" * 65}')
    print(f'  汇总:')
    for sp in ['train', 'val', 'test']:
        print(f'    {sp:5s}: {stats[sp]} runs')
    print(f'    total: {sum(stats.values())} runs')
    print(f'{"=" * 65}')

    return result


def _make_row(rec, split):
    """构造输出行"""
    return {
        'run_name': rec['run_name'],
        'split': split,
        'junction_id': rec['junction_id'],
        'turn_dir': rec['turn_dir'],
        'pre_turn_ms_available': rec['pre_turn_ms_available'],
        'delay_bucket': rec['delay_bucket'],
        'has_follow': rec['has_follow'],
        'has_all_4_phases': rec['has_all_4_phases'],
    }


# ============================================================================
# 输出
# ============================================================================

OUT_FIELDS = [
    'run_name', 'split', 'junction_id', 'turn_dir',
    'pre_turn_ms_available', 'delay_bucket',
    'has_follow', 'has_all_4_phases',
]


def write_fixed_split(out_csv, rows):
    """写出固定划分 CSV"""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='根据 stage4_run_manifest.csv 生成固定 train/val/test 划分',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--manifest_csv', type=str, required=True,
                        help='stage4_run_manifest.csv 路径')
    parser.add_argument('--out_csv', type=str,
                        default='./data/stage4_fixed_split.csv',
                        help='输出固定划分 CSV 路径')

    args = parser.parse_args()

    if not os.path.isfile(args.manifest_csv):
        print(f'✗ manifest_csv 不存在: {args.manifest_csv}')
        return

    print(f'  输入: {os.path.abspath(args.manifest_csv)}')
    print(f'  输出: {os.path.abspath(args.out_csv)}')

    records = load_manifest(args.manifest_csv)
    if not records:
        print(f'✗ manifest 为空')
        return

    rows = generate_fixed_split(records)
    write_fixed_split(args.out_csv, rows)
    print(f'\n  [✓] {args.out_csv} ({len(rows)} 条)')

    print(f'\n  下一步:')
    print(f'    python scripts/split_corridor_runs.py \\')
    print(f'        --src_root <your_src_root> \\')
    print(f'        --dst_root <your_dst_root> \\')
    print(f'        --fixed_split_csv {args.out_csv}')


if __name__ == '__main__':
    main()
