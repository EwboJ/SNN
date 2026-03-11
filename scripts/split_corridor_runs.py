"""
走廊导航 run 级别数据集划分 (v2)
=====================================
按 (junction_id, turn_dir) 分层，将 run 目录划分为 train / val / test。

数据组织:
    4 个路口 × 2 方向 × 7 次采集 = 56 runs
    每个 run 包含 images/ + labels.csv + meta.json

划分策略:
    exact 模式: 每组精确指定 train/val/test 数量 (默认 5+1+1)
    ratio 模式: 按比例随机划分 (兼容旧版)

用法:
    # 精确模式 (推荐)
    python scripts/split_corridor_runs.py \\
        --src_root ./data/corridor_balanced \\
        --dst_root ./data/corridor \\
        --split_mode exact \\
        --train_per_group 5 --val_per_group 1 --test_per_group 1

    # 比例模式
    python scripts/split_corridor_runs.py \\
        --src_root ./data/corridor_balanced \\
        --dst_root ./data/corridor \\
        --split_mode ratio \\
        --val_ratio 0.15 --test_ratio 0.15

    # 使用 manifest
    python scripts/split_corridor_runs.py \\
        --src_root ./data/corridor_balanced \\
        --dst_root ./data/corridor \\
        --manifest runs_manifest.csv \\
        --split_mode exact

输出:
    <dst_root>/
      train/<run_name>/...
      val/<run_name>/...
      test/<run_name>/...
      split_manifest.csv
      split_summary.json
"""

import os
import sys
import re
import csv
import json
import math
import shutil
import random
import argparse
from collections import defaultdict, OrderedDict


# ============================================================================
# 基础工具 (保留自 v1)
# ============================================================================

def find_valid_runs(src_root, min_frames=0):
    """
    扫描 src_root 下所有有效 run 目录。
    有效条件: 含 images/ + labels.csv。

    Returns:
        list of dict: [{name, path, frame_count, valid_frames}, ...]
    """
    runs = []
    for name in sorted(os.listdir(src_root)):
        rp = os.path.join(src_root, name)
        if not os.path.isdir(rp):
            continue
        img_dir = os.path.join(rp, 'images')
        lbl_csv = os.path.join(rp, 'labels.csv')
        if not os.path.isdir(img_dir) or not os.path.isfile(lbl_csv):
            continue

        # 统计帧数
        frame_count = 0
        valid_frames = 0
        meta_path = os.path.join(rp, 'meta.json')
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                frame_count = meta.get('total_frames', 0)
                valid_frames = meta.get('valid_frames', frame_count)
            except Exception:
                pass

        if frame_count == 0:
            try:
                with open(lbl_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        frame_count += 1
                        if int(row.get('valid', '1')) == 1:
                            valid_frames += 1
            except Exception:
                pass

        if frame_count < min_frames:
            print(f"  [跳过] {name}: {frame_count} 帧 < min_frames={min_frames}")
            continue

        runs.append({
            'name': name,
            'path': rp,
            'frame_count': frame_count,
            'valid_frames': valid_frames,
        })

    return runs


def get_action_dist(run_path):
    """从 meta.json 或 labels.csv 读取动作分布"""
    meta_path = os.path.join(run_path, 'meta.json')
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            return meta.get('action_distribution', {})
        except Exception:
            pass
    # fallback
    dist = {}
    lbl_csv = os.path.join(run_path, 'labels.csv')
    try:
        with open(lbl_csv, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                act = row.get('action_name', 'unknown')
                dist[act] = dist.get(act, 0) + 1
    except Exception:
        pass
    return dist


def copy_run(src, dst, mode):
    """复制或创建符号链接"""
    if mode == 'symlink':
        src_abs = os.path.abspath(src)
        os.symlink(src_abs, dst, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


# ============================================================================
# 目录名解析
# ============================================================================

# 新格式: J1_left_r03
_RE_NEW = re.compile(
    r'^J(?P<jid>\d+)_(?P<turn>left|right)_r(?P<rep>\d+)$', re.IGNORECASE)

# 旧格式: left1_bag3, right2_bag7
_RE_OLD = re.compile(
    r'^(?P<turn>left|right)(?P<jid>\d+)_bag(?P<rep>\d+)$', re.IGNORECASE)


def parse_run_name(name):
    """
    从目录名解析 junction_id, turn_dir, rep_id。

    支持格式:
      J1_left_r03  → (1, 'left', 3)
      left1_bag3   → (1, 'left', 3)

    Returns:
        (junction_id: int, turn_dir: str, rep_id: int) or None
    """
    m = _RE_NEW.match(name)
    if m:
        return int(m.group('jid')), m.group('turn').lower(), int(m.group('rep'))

    m = _RE_OLD.match(name)
    if m:
        return int(m.group('jid')), m.group('turn').lower(), int(m.group('rep'))

    return None


def load_manifest(manifest_path):
    """
    从 runs_manifest.csv 加载元数据。

    CSV 格式: run_name, junction_id, turn_dir, rep_id [, ...]

    Returns:
        dict: {run_name: {junction_id, turn_dir, rep_id}}
    """
    result = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['run_name'].strip()
            result[name] = {
                'junction_id': int(row['junction_id']),
                'turn_dir': row['turn_dir'].strip().lower(),
                'rep_id': int(row.get('rep_id', 0)),
            }
    return result


# ============================================================================
# 分组与划分
# ============================================================================

def build_groups(runs, manifest, group_keys):
    """
    将 runs 按 group_keys 分组。

    Args:
        runs: find_valid_runs 结果
        manifest: {run_name: {junction_id, turn_dir, rep_id}} or None
        group_keys: 如 ['junction_id', 'turn_dir']

    Returns:
        groups: {group_key_tuple: [run_info, ...]}
        所有 run_info 会增加 junction_id, turn_dir, rep_id 字段
    """
    groups = defaultdict(list)
    parse_fail = []

    for run in runs:
        name = run['name']
        meta = None

        if manifest and name in manifest:
            meta = manifest[name]
        else:
            parsed = parse_run_name(name)
            if parsed:
                meta = {
                    'junction_id': parsed[0],
                    'turn_dir': parsed[1],
                    'rep_id': parsed[2],
                }

        if meta is None:
            parse_fail.append(name)
            continue

        run.update(meta)
        key = tuple(run[k] for k in group_keys)
        groups[key].append(run)

    if parse_fail:
        print(f"\n  ⚠ 以下 {len(parse_fail)} 个 run 无法解析分组，已跳过:")
        for n in parse_fail:
            print(f"    - {n}")

    return dict(groups)


def split_exact(groups, train_n, val_n, test_n, seed):
    """
    精确模式: 每组取 train_n + val_n + test_n 个 run。

    Returns:
        (train_runs, val_runs, test_runs)    -- 三个 list
    """
    rng = random.Random(seed)
    train_runs, val_runs, test_runs = [], [], []
    errors = []

    for key in sorted(groups.keys()):
        grp = groups[key]
        required = train_n + val_n + test_n
        if len(grp) < required:
            errors.append(
                f"  组 {key}: 需要 {required} 个 run，实际只有 {len(grp)} 个")
            continue

        shuffled = list(grp)
        rng.shuffle(shuffled)

        test_runs.extend(shuffled[:test_n])
        val_runs.extend(shuffled[test_n:test_n + val_n])
        train_runs.extend(shuffled[test_n + val_n:test_n + val_n + train_n])

    if errors:
        print("\n  ✗ 以下组的 run 数量不足:")
        for e in errors:
            print(e)
        print(f"\n  请检查数据或降低 train/val/test 的数量要求。")
        sys.exit(1)

    return train_runs, val_runs, test_runs


def split_ratio(groups, val_ratio, test_ratio, seed):
    """
    比例模式: 按组内随机划分。

    Returns:
        (train_runs, val_runs, test_runs)
    """
    rng = random.Random(seed)
    train_runs, val_runs, test_runs = [], [], []

    for key in sorted(groups.keys()):
        grp = list(groups[key])
        rng.shuffle(grp)

        n = len(grp)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
        n_train = n - n_test - n_val

        if n_train < 1:
            n_train = 1
            leftover = n - 1
            n_test = max(1, round(leftover * test_ratio / (test_ratio + val_ratio)))
            n_val = leftover - n_test

        test_runs.extend(grp[:n_test])
        val_runs.extend(grp[n_test:n_test + n_val])
        train_runs.extend(grp[n_test + n_val:])

    return train_runs, val_runs, test_runs


# ============================================================================
# 统计与输出
# ============================================================================

def compute_split_stats(runs, group_keys):
    """计算单个 split 的统计"""
    total_frames = sum(r['frame_count'] for r in runs)
    total_valid = sum(r.get('valid_frames', r['frame_count']) for r in runs)

    action_dist = defaultdict(int)
    for r in runs:
        ad = get_action_dist(r['path'])
        for k, v in ad.items():
            action_dist[k] += v

    group_dist = defaultdict(int)
    for r in runs:
        key = tuple(r.get(k, '?') for k in group_keys)
        group_dist[key] += 1

    return {
        'n_runs': len(runs),
        'total_frames': total_frames,
        'total_valid': total_valid,
        'action_distribution': dict(action_dist),
        'group_distribution': {str(k): v for k, v in sorted(group_dist.items())},
        'run_names': sorted(r['name'] for r in runs),
    }


def print_split_info(name, stats, group_keys):
    """打印单个 split 的信息"""
    print(f"\n  ── {name} ──")
    print(f"    Run 数:  {stats['n_runs']}")
    print(f"    总帧数:  {stats['total_frames']}  "
          f"(有效: {stats['total_valid']})")
    ad = stats['action_distribution']
    if ad:
        ad_str = ', '.join(f'{k}:{v}' for k, v in sorted(ad.items()))
        print(f"    动作分布: {ad_str}")
    gd = stats['group_distribution']
    if gd:
        print(f"    按组分布:")
        for gk, gv in sorted(gd.items()):
            print(f"      {gk}: {gv} runs")


def save_split_manifest(dst_root, train_runs, val_runs, test_runs):
    """保存 split_manifest.csv"""
    path = os.path.join(dst_root, 'split_manifest.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_name', 'split', 'junction_id', 'turn_dir',
            'rep_id', 'frame_count', 'valid_frames'
        ])
        for split_name, runs in [
            ('train', train_runs), ('val', val_runs), ('test', test_runs)
        ]:
            for r in sorted(runs, key=lambda x: x['name']):
                writer.writerow([
                    r['name'], split_name,
                    r.get('junction_id', ''),
                    r.get('turn_dir', ''),
                    r.get('rep_id', ''),
                    r['frame_count'],
                    r.get('valid_frames', ''),
                ])
    print(f"  [✓] {path}")


def save_split_summary(dst_root, config, train_stats, val_stats, test_stats):
    """保存 split_summary.json"""
    path = os.path.join(dst_root, 'split_summary.json')
    summary = {
        'config': config,
        'train': train_stats,
        'val': val_stats,
        'test': test_stats,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [✓] {path}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='走廊导航 run 级别 train/val/test 划分 (v2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 路径
    parser.add_argument('--src_root', type=str, default='./data/corridor_balanced',
                        help='原始 run 目录的根路径')
    parser.add_argument('--dst_root', type=str, default='./data/corridor',
                        help='输出目录 (创建 train/ val/ test/)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='可选 runs_manifest.csv 路径')

    # 划分模式
    parser.add_argument('--split_mode', type=str, default='exact',
                        choices=['exact', 'ratio'],
                        help='划分模式: exact=每组精确数量, ratio=按比例')
    parser.add_argument('--group_by', type=str, default='junction_id,turn_dir',
                        help='分组键 (逗号分隔)')

    # exact 模式参数
    parser.add_argument('--train_per_group', type=int, default=5,
                        help='[exact] 每组 train 的 run 数')
    parser.add_argument('--val_per_group', type=int, default=1,
                        help='[exact] 每组 val 的 run 数')
    parser.add_argument('--test_per_group', type=int, default=1,
                        help='[exact] 每组 test 的 run 数')

    # ratio 模式参数
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='[ratio] val 比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='[ratio] test 比例')

    # 过滤
    parser.add_argument('--min_frames', type=int, default=10,
                        help='最小帧数阈值 (低于此值的 run 被跳过)')
    parser.add_argument('--exclude', nargs='*', default=[],
                        help='排除的 run 名称')

    # 复制
    parser.add_argument('--copy_mode', type=str, default='symlink',
                        choices=['copy', 'symlink'],
                        help='复制方式 (推荐 symlink 节省磁盘)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--force', action='store_true',
                        help='覆盖已有目标目录')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅预览，不实际复制')

    args = parser.parse_args()
    group_keys = [k.strip() for k in args.group_by.split(',')]
    exclude_set = set(args.exclude) if args.exclude else set()

    # ======================== Banner ========================
    print('=' * 72)
    print('  走廊导航数据集划分 (v2)')
    print('=' * 72)
    print(f'  源目录:       {os.path.abspath(args.src_root)}')
    print(f'  目标目录:     {os.path.abspath(args.dst_root)}')
    print(f'  划分模式:     {args.split_mode}')
    print(f'  分组键:       {group_keys}')
    if args.split_mode == 'exact':
        print(f'  每组分配:     train={args.train_per_group}, '
              f'val={args.val_per_group}, test={args.test_per_group}')
    else:
        print(f'  比例:         val={args.val_ratio}, test={args.test_ratio}')
    print(f'  复制方式:     {args.copy_mode}')
    print(f'  种子:         {args.seed}')
    if exclude_set:
        print(f'  排除:         {exclude_set}')
    print('=' * 72)

    # ======================== [1] 扫描 ========================
    print(f'\n[1/5] 扫描有效 run...')

    if not os.path.isdir(args.src_root):
        print(f"  ✗ 源目录不存在: {args.src_root}")
        sys.exit(1)

    all_runs = find_valid_runs(args.src_root, min_frames=args.min_frames)

    # 排除
    if exclude_set:
        before = len(all_runs)
        all_runs = [r for r in all_runs if r['name'] not in exclude_set]
        print(f"  排除 {before - len(all_runs)} 个 run")

    if not all_runs:
        print("  ✗ 未找到有效 run!")
        sys.exit(1)

    print(f"  找到 {len(all_runs)} 个有效 run, "
          f"总帧数 {sum(r['frame_count'] for r in all_runs)}")

    # ======================== [2] 分组 ========================
    print(f'\n[2/5] 按 {group_keys} 分组...')

    manifest = None
    if args.manifest:
        if not os.path.isfile(args.manifest):
            print(f"  ✗ manifest 文件不存在: {args.manifest}")
            sys.exit(1)
        manifest = load_manifest(args.manifest)
        print(f"  从 manifest 加载 {len(manifest)} 条记录")

    groups = build_groups(all_runs, manifest, group_keys)

    if not groups:
        print("  ✗ 无法解析任何 run 的分组信息!")
        print("  请确保目录名格式为 J{id}_{left|right}_r{rep} "
              "或 {left|right}{id}_bag{rep}")
        print("  或提供 --manifest 文件")
        sys.exit(1)

    print(f"  共 {len(groups)} 个组:")
    for key in sorted(groups.keys()):
        grp = groups[key]
        frames = sum(r['frame_count'] for r in grp)
        names = ', '.join(r['name'] for r in sorted(grp, key=lambda x: x['name']))
        print(f"    {key}: {len(grp)} runs, {frames} 帧  [{names}]")

    # ======================== [3] 划分 ========================
    print(f'\n[3/5] 执行 {args.split_mode} 划分...')

    if args.split_mode == 'exact':
        train_runs, val_runs, test_runs = split_exact(
            groups,
            train_n=args.train_per_group,
            val_n=args.val_per_group,
            test_n=args.test_per_group,
            seed=args.seed)
    else:
        train_runs, val_runs, test_runs = split_ratio(
            groups,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed)

    # 完整性检查: 同一 run 不能出现在多个 split
    all_names = set()
    for split_name, runs in [('train', train_runs), ('val', val_runs),
                              ('test', test_runs)]:
        for r in runs:
            if r['name'] in all_names:
                print(f"  ✗ 致命错误: run '{r['name']}' 出现在多个 split 中!")
                sys.exit(1)
            all_names.add(r['name'])

    train_stats = compute_split_stats(train_runs, group_keys)
    val_stats = compute_split_stats(val_runs, group_keys)
    test_stats = compute_split_stats(test_runs, group_keys)

    print_split_info('TRAIN', train_stats, group_keys)
    print_split_info('VAL', val_stats, group_keys)
    print_split_info('TEST', test_stats, group_keys)

    # ======================== [4] 复制 ========================
    if args.dry_run:
        print(f'\n[4/5] [DRY RUN] 跳过复制')
    else:
        print(f'\n[4/5] 复制 run (mode={args.copy_mode})...')

        # 检查目标目录
        for sub in ['train', 'val', 'test']:
            sub_dir = os.path.join(args.dst_root, sub)
            if os.path.exists(sub_dir):
                if not args.force:
                    ans = input(f"  ⚠ {sub_dir} 已存在，覆盖? [y/N] ").strip().lower()
                    if ans not in ('y', 'yes'):
                        print("  已取消。")
                        sys.exit(0)
                shutil.rmtree(sub_dir)

        for split_name, runs in [('train', train_runs), ('val', val_runs),
                                  ('test', test_runs)]:
            split_dir = os.path.join(args.dst_root, split_name)
            os.makedirs(split_dir, exist_ok=True)
            for r in runs:
                dst = os.path.join(split_dir, r['name'])
                copy_run(r['path'], dst, args.copy_mode)
                print(f"    [{split_name:5s}] {r['name']}")

    # ======================== [5] 保存元数据 ========================
    print(f'\n[5/5] 保存元数据...')
    os.makedirs(args.dst_root, exist_ok=True)

    config_out = {
        'src_root': args.src_root,
        'dst_root': args.dst_root,
        'split_mode': args.split_mode,
        'group_by': group_keys,
        'seed': args.seed,
        'copy_mode': args.copy_mode,
        'exclude': list(exclude_set),
        'min_frames': args.min_frames,
    }
    if args.split_mode == 'exact':
        config_out['train_per_group'] = args.train_per_group
        config_out['val_per_group'] = args.val_per_group
        config_out['test_per_group'] = args.test_per_group
    else:
        config_out['val_ratio'] = args.val_ratio
        config_out['test_ratio'] = args.test_ratio

    save_split_manifest(args.dst_root, train_runs, val_runs, test_runs)
    save_split_summary(args.dst_root, config_out, train_stats, val_stats,
                       test_stats)

    # ======================== 总结 ========================
    total = train_stats['n_runs'] + val_stats['n_runs'] + test_stats['n_runs']
    total_f = (train_stats['total_frames'] + val_stats['total_frames'] +
               test_stats['total_frames'])

    print(f'\n{"=" * 72}')
    print(f'  划分完成!')
    print(f'{"=" * 72}')
    print(f'  Train:  {train_stats["n_runs"]:3d} runs  '
          f'{train_stats["total_frames"]:6d} 帧  '
          f'({100*train_stats["n_runs"]/total:.0f}%)')
    print(f'  Val:    {val_stats["n_runs"]:3d} runs  '
          f'{val_stats["total_frames"]:6d} 帧  '
          f'({100*val_stats["n_runs"]/total:.0f}%)')
    print(f'  Test:   {test_stats["n_runs"]:3d} runs  '
          f'{test_stats["total_frames"]:6d} 帧  '
          f'({100*test_stats["n_runs"]/total:.0f}%)')
    print(f'  Total:  {total:3d} runs  {total_f:6d} 帧')

    # 分组分布表
    print(f'\n  分组在各 split 中的分布:')
    all_group_keys = sorted(set(
        list(train_stats['group_distribution'].keys()) +
        list(val_stats['group_distribution'].keys()) +
        list(test_stats['group_distribution'].keys())))

    print(f'  {"Group":25s} {"Train":>6s} {"Val":>6s} {"Test":>6s} {"Total":>6s}')
    print(f'  {"─" * 55}')
    for gk in all_group_keys:
        t = train_stats['group_distribution'].get(gk, 0)
        v = val_stats['group_distribution'].get(gk, 0)
        te = test_stats['group_distribution'].get(gk, 0)
        s = t + v + te
        print(f'  {gk:25s} {t:6d} {v:6d} {te:6d} {s:6d}')

    # 缺失检查
    missing = []
    for gk in all_group_keys:
        t = train_stats['group_distribution'].get(gk, 0)
        v = val_stats['group_distribution'].get(gk, 0)
        te = test_stats['group_distribution'].get(gk, 0)
        if t == 0:
            missing.append(f"  {gk}: 0 train runs")
        if te == 0:
            missing.append(f"  {gk}: 0 test runs")

    if missing:
        print(f'\n  ⚠ 警告 — 以下组存在缺失:')
        for m in missing:
            print(f'    {m}')
    else:
        print(f'\n  ✓ 所有组在 train/val/test 中均有覆盖')

    print(f'{"=" * 72}')

    if args.dry_run:
        print(f'  [DRY RUN] 以上为预览，未实际复制文件。')
        print(f'  去掉 --dry_run 后重新运行即可执行。')
    else:
        print(f'  输出目录: {os.path.abspath(args.dst_root)}')
        print(f'\n  下一步训练命令:')
        print(f'    python train.py --dataset corridor '
              f'--corridor_root {args.dst_root} '
              f'--mode discrete --action_set 3 -T 4 '
              f'--class_balance weighted_sampler -b 32')
    print(f'{"=" * 72}')


if __name__ == '__main__':
    main()
