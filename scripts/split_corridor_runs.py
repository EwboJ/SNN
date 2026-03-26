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
    fixed 模式: 从 CSV 文件直接指定每个 run 的 split (不随机)

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

    # 固定划分 (从 CSV 指定)
    python scripts/split_corridor_runs.py \\
        --src_root ./data/corridor_balanced \\
        --dst_root ./data/corridor \\
        --fixed_split_csv ./data/fixed_split.csv

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
    从 manifest CSV 加载元数据。

    支持两种格式:
      - junction 格式: run_name, junction_id, turn_dir, rep_id [, ...]
      - straight_keep 格式: run_name, segment_id, direction, station_id,
                            offset_cm, yaw_deg, target_speed_mps, rep_id

    自动读取 CSV 全部字段，保留原始列名和值。
    数值型字段尝试自动转换 (int -> float -> str)。

    Returns:
        tuple: (manifest_dict, field_names)
          manifest_dict: {run_name: {field: value, ...}}
          field_names: CSV 中除 run_name 外的所有列名 (保持原始顺序)
    """
    result = {}
    field_names = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            # 保留除 run_name 外的所有列名
            field_names = [fn.strip() for fn in reader.fieldnames
                          if fn.strip() != 'run_name']
        for row in reader:
            name = row['run_name'].strip()
            record = {}
            for fn in field_names:
                raw_val = row.get(fn, '').strip()
                # 尝试自动类型转换
                record[fn] = _auto_convert(raw_val)
            result[name] = record
    return result, field_names


def _auto_convert(val_str):
    """尝试将字符串转为 int / float，失败则保留 str"""
    if val_str == '':
        return val_str
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str.lower() if val_str.isalpha() else val_str


# ============================================================================
# 分组与划分
# ============================================================================

def build_groups(runs, manifest, group_keys, manifest_fields=None):
    """
    将 runs 按 group_keys 分组。

    Args:
        runs: find_valid_runs 结果
        manifest: {run_name: {field: value}} or None
        group_keys: 如 ['junction_id', 'turn_dir']
                    或 ['segment_id', 'direction', 'station_id']
        manifest_fields: manifest 列名列表 (可选，用于检查 group_keys 合法性)

    Returns:
        groups: {group_key_tuple: [run_info, ...]}
        所有 run_info 会增加 manifest 中对应的字段
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
# 固定划分 (fixed split)
# ============================================================================

def load_fixed_split_csv(csv_path):
    """
    读取固定划分 CSV 文件。

    CSV 至少包含 run_name, split 两列。
    split 只能为 train / val / test。

    Returns:
        dict: {run_name: split_str}
    """
    mapping = {}
    valid_splits = {'train', 'val', 'test'}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"fixed_split_csv 为空: {csv_path}")
        # 检查必要列
        for col in ['run_name', 'split']:
            if col not in reader.fieldnames:
                raise ValueError(
                    f"fixed_split_csv 缺少 '{col}' 列: {csv_path}\n"
                    f"  实际列名: {reader.fieldnames}")

        for i, row in enumerate(reader, 1):
            name = row['run_name'].strip()
            split = row['split'].strip().lower()
            if not name:
                print(f"  ⚠ fixed_split_csv 第 {i} 行: run_name 为空，跳过")
                continue
            if split not in valid_splits:
                raise ValueError(
                    f"fixed_split_csv 第 {i} 行: "
                    f"split='{split}' 无效 (只能为 train/val/test)")
            if name in mapping:
                raise ValueError(
                    f"fixed_split_csv 中 run '{name}' 重复出现!\n"
                    f"  第一次: split={mapping[name]}\n"
                    f"  第二次: split={split}")
            mapping[name] = split

    return mapping


def apply_fixed_split(all_runs, fixed_mapping):
    """
    按固定映射分配 train/val/test。

    一致性检查:
      1. run 不允许重复出现在多个 split
      2. 所有有效 run 必须被分配
      3. fixed_split_csv 中的 run_name 必须都存在于 src_root

    Returns:
        (train_runs, val_runs, test_runs)
    """
    train_runs, val_runs, test_runs = [], [], []
    run_name_set = {r['name'] for r in all_runs}

    # 检查 CSV 中的 run 是否都存在
    csv_names = set(fixed_mapping.keys())
    missing_in_src = csv_names - run_name_set
    if missing_in_src:
        print(f"\n  ⚠ fixed_split_csv 中以下 run 在 src_root 中不存在:")
        for n in sorted(missing_in_src):
            print(f"    - {n}")
        raise RuntimeError(
            f"fixed_split_csv 中 {len(missing_in_src)} 个 run "
            f"在 src_root 中不存在")

    # 检查所有有效 run 是否都被分配
    unassigned = run_name_set - csv_names
    if unassigned:
        print(f"\n  ⚠ 以下 {len(unassigned)} 个有效 run 未在 "
              f"fixed_split_csv 中出现:")
        for n in sorted(unassigned):
            print(f"    - {n}")
        raise RuntimeError(
            f"{len(unassigned)} 个有效 run 未被分配到任何 split")

    # 分配
    for run in all_runs:
        split = fixed_mapping[run['name']]
        if split == 'train':
            train_runs.append(run)
        elif split == 'val':
            val_runs.append(run)
        elif split == 'test':
            test_runs.append(run)

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


def save_split_manifest(dst_root, train_runs, val_runs, test_runs,
                        manifest_fields=None):
    """
    保存 split_manifest.csv。
    动态包含 manifest 中的所有字段列。
    """
    path = os.path.join(dst_root, 'split_manifest.csv')

    # 确定要输出的 manifest 字段
    if manifest_fields:
        extra_cols = manifest_fields
    else:
        # 回退到 junction 默认字段
        extra_cols = ['junction_id', 'turn_dir', 'rep_id']

    header = ['run_name', 'split'] + extra_cols + ['frame_count', 'valid_frames']

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for split_name, runs in [
            ('train', train_runs), ('val', val_runs), ('test', test_runs)
        ]:
            for r in sorted(runs, key=lambda x: x['name']):
                row = [r['name'], split_name]
                for col in extra_cols:
                    row.append(r.get(col, ''))
                row.append(r['frame_count'])
                row.append(r.get('valid_frames', ''))
                writer.writerow(row)
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
# 可调用入口 (供 pipeline 调用)
# ============================================================================

def run_split(src_root, dst_root, split_mode='exact', group_by='junction_id,turn_dir',
              train_per_group=5, val_per_group=1, test_per_group=1,
              val_ratio=0.15, test_ratio=0.15,
              min_frames=10, exclude=None, manifest_path=None,
              fixed_split_csv=None,
              copy_mode='symlink', seed=42, force=False, dry_run=False):
    """
    走廊导航 run 级别 train/val/test 划分核心函数 (供 pipeline 调用)。

    Returns:
        dict: split_summary
    """
    group_keys = [k.strip() for k in group_by.split(',')]
    exclude_set = set(exclude) if exclude else set()

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"源目录不存在: {src_root}")

    # 判断是否使用固定划分
    use_fixed = fixed_split_csv is not None

    # ======================== Banner ========================
    print('=' * 72)
    print('  走廊导航数据集划分 (v2)')
    print('=' * 72)
    print(f'  源目录:       {os.path.abspath(src_root)}')
    print(f'  目标目录:     {os.path.abspath(dst_root)}')
    if use_fixed:
        print(f'  划分模式:     fixed (从 CSV 指定)')
        print(f'  固定划分文件: {os.path.abspath(fixed_split_csv)}')
    else:
        print(f'  划分模式:     {split_mode}')
        print(f'  分组键:       {group_keys}')
        if split_mode == 'exact':
            print(f'  每组分配:     train={train_per_group}, '
                  f'val={val_per_group}, test={test_per_group}')
        else:
            print(f'  比例:         val={val_ratio}, test={test_ratio}')
        print(f'  种子:         {seed}')
    print(f'  复制方式:     {copy_mode}')
    if exclude_set:
        print(f'  排除:         {exclude_set}')
    print('=' * 72)

    # ======================== [1] 扫描 ========================
    print(f'\n[1/5] 扫描有效 run...')

    all_runs = find_valid_runs(src_root, min_frames=min_frames)

    if exclude_set:
        before = len(all_runs)
        all_runs = [r for r in all_runs if r['name'] not in exclude_set]
        print(f"  排除 {before - len(all_runs)} 个 run")

    if not all_runs:
        raise RuntimeError("未找到有效 run!")

    print(f"  找到 {len(all_runs)} 个有效 run, "
          f"总帧数 {sum(r['frame_count'] for r in all_runs)}")

    # ======================== [2] 分组 / 固定划分加载 ========================
    manifest = None
    manifest_fields = None   # manifest 中除 run_name 外的所有列名
    fixed_mapping = None

    if use_fixed:
        print(f'\n[2/5] 加载固定划分文件...')
        if not os.path.isfile(fixed_split_csv):
            raise FileNotFoundError(
                f"fixed_split_csv 文件不存在: {fixed_split_csv}")
        fixed_mapping = load_fixed_split_csv(fixed_split_csv)
        print(f"  加载 {len(fixed_mapping)} 条划分记录")
        # 统计各 split 数量
        from collections import Counter
        split_counts = Counter(fixed_mapping.values())
        for sp in ['train', 'val', 'test']:
            print(f"    {sp}: {split_counts.get(sp, 0)} runs")
    else:
        print(f'\n[2/5] 按 {group_keys} 分组...')
        if manifest_path:
            if not os.path.isfile(manifest_path):
                raise FileNotFoundError(
                    f"manifest 文件不存在: {manifest_path}")
            manifest, manifest_fields = load_manifest(manifest_path)
            print(f"  从 manifest 加载 {len(manifest)} 条记录")
            print(f"  manifest 字段: {manifest_fields}")
            # 检查 group_keys 是否包含在 manifest 字段中
            missing_keys = [k for k in group_keys
                            if k not in manifest_fields]
            if missing_keys:
                print(f"  ⚠ 分组键 {missing_keys} 不在 manifest 字段中!")
                print(f"    可用字段: {manifest_fields}")
                print(f"    将尝试从目录名解析缺失字段")

    # ======================== [3] 划分 ========================
    if use_fixed:
        print(f'\n[3/5] 执行固定划分 (fixed)...')

        # 如有 manifest，先加载以便后续写 split_manifest.csv 带完整字段
        if manifest_path and os.path.isfile(manifest_path):
            manifest, manifest_fields = load_manifest(manifest_path)
            # 把 manifest 信息合并到 run 上
            for run in all_runs:
                if manifest and run['name'] in manifest:
                    run.update(manifest[run['name']])
        else:
            # 无 manifest 时尝试从目录名解析
            for run in all_runs:
                parsed = parse_run_name(run['name'])
                if parsed:
                    run['junction_id'] = parsed[0]
                    run['turn_dir'] = parsed[1]
                    run['rep_id'] = parsed[2]

        train_runs, val_runs, test_runs = apply_fixed_split(
            all_runs, fixed_mapping)
        print(f"  [✓] 固定划分完成")
    else:
        groups = build_groups(all_runs, manifest, group_keys,
                              manifest_fields)

        if not groups:
            raise RuntimeError(
                "无法解析任何 run 的分组信息! "
                "请确保目录名格式为 J{id}_{left|right}_r{rep} "
                "或 {left|right}{id}_bag{rep}，或提供 --manifest 文件")

        print(f"  共 {len(groups)} 个组:")
        for key in sorted(groups.keys()):
            grp = groups[key]
            frames = sum(r['frame_count'] for r in grp)
            names = ', '.join(
                r['name'] for r in sorted(grp, key=lambda x: x['name']))
            print(f"    {key}: {len(grp)} runs, {frames} 帧  [{names}]")

        print(f'\n[3/5] 执行 {split_mode} 划分...')

        if split_mode == 'exact':
            train_runs, val_runs, test_runs = split_exact(
                groups,
                train_n=train_per_group,
                val_n=val_per_group,
                test_n=test_per_group,
                seed=seed)
        else:
            train_runs, val_runs, test_runs = split_ratio(
                groups,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed)

    # 完整性检查
    all_names = set()
    for sn, runs in [('train', train_runs), ('val', val_runs),
                     ('test', test_runs)]:
        for r in runs:
            if r['name'] in all_names:
                raise RuntimeError(f"致命错误: run '{r['name']}' 出现在多个 split 中!")
            all_names.add(r['name'])

    train_stats = compute_split_stats(train_runs, group_keys)
    val_stats = compute_split_stats(val_runs, group_keys)
    test_stats = compute_split_stats(test_runs, group_keys)

    print_split_info('TRAIN', train_stats, group_keys)
    print_split_info('VAL', val_stats, group_keys)
    print_split_info('TEST', test_stats, group_keys)

    # ======================== [4] 复制 ========================
    if dry_run:
        print(f'\n[4/5] [DRY RUN] 跳过复制')
    else:
        print(f'\n[4/5] 复制 run (mode={copy_mode})...')

        for sub in ['train', 'val', 'test']:
            sub_dir = os.path.join(dst_root, sub)
            if os.path.exists(sub_dir):
                if not force:
                    raise FileExistsError(
                        f"{sub_dir} 已存在  (使用 --force 覆盖)")
                shutil.rmtree(sub_dir)

        for sn, runs in [('train', train_runs), ('val', val_runs),
                         ('test', test_runs)]:
            split_dir = os.path.join(dst_root, sn)
            os.makedirs(split_dir, exist_ok=True)
            for r in runs:
                dst = os.path.join(split_dir, r['name'])
                copy_run(r['path'], dst, copy_mode)
                print(f"    [{sn:5s}] {r['name']}")

    # ======================== [5] 保存元数据 ========================
    print(f'\n[5/5] 保存元数据...')
    os.makedirs(dst_root, exist_ok=True)

    config_out = {
        'src_root': src_root,
        'dst_root': dst_root,
        'split_mode': 'fixed' if use_fixed else split_mode,
        'group_by': group_keys,
        'seed': seed,
        'copy_mode': copy_mode,
        'exclude': list(exclude_set),
        'min_frames': min_frames,
    }
    if use_fixed:
        config_out['fixed_split_csv'] = os.path.abspath(fixed_split_csv)
    elif split_mode == 'exact':
        config_out['train_per_group'] = train_per_group
        config_out['val_per_group'] = val_per_group
        config_out['test_per_group'] = test_per_group
    else:
        config_out['val_ratio'] = val_ratio
        config_out['test_ratio'] = test_ratio

    save_split_manifest(dst_root, train_runs, val_runs, test_runs,
                        manifest_fields=manifest_fields)
    save_split_summary(dst_root, config_out, train_stats, val_stats,
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

    if dry_run:
        print(f'  [DRY RUN] 以上为预览，未实际复制文件。')
    else:
        print(f'  输出目录: {os.path.abspath(dst_root)}')

    summary = {
        'config': config_out,
        'train': train_stats,
        'val': val_stats,
        'test': test_stats,
    }
    return summary


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='走廊导航 run 级别 train/val/test 划分 (v2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_root', type=str, default='./data/corridor_balanced',
                        help='原始 run 目录的根路径')
    parser.add_argument('--dst_root', type=str, default='./data/corridor',
                        help='输出目录 (创建 train/ val/ test/)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='可选 runs_manifest.csv 路径')
    parser.add_argument('--split_mode', type=str, default='exact',
                        choices=['exact', 'ratio'],
                        help='划分模式')
    parser.add_argument('--group_by', type=str, default='junction_id,turn_dir',
                        help='分组键 (逗号分隔)')
    parser.add_argument('--train_per_group', type=int, default=5)
    parser.add_argument('--val_per_group', type=int, default=1)
    parser.add_argument('--test_per_group', type=int, default=1)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--min_frames', type=int, default=10)
    parser.add_argument('--exclude', nargs='*', default=[])
    parser.add_argument('--copy_mode', type=str, default='symlink',
                        choices=['copy', 'symlink'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true',
                        help='覆盖已有目标目录')
    parser.add_argument('--fixed_split_csv', type=str, default=None,
                        help='固定划分 CSV 文件路径 '
                             '(至少含 run_name, split 列)')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅预览，不实际复制')

    args = parser.parse_args()

    run_split(
        src_root=args.src_root,
        dst_root=args.dst_root,
        split_mode=args.split_mode,
        group_by=args.group_by,
        train_per_group=args.train_per_group,
        val_per_group=args.val_per_group,
        test_per_group=args.test_per_group,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_frames=args.min_frames,
        exclude=args.exclude,
        manifest_path=args.manifest,
        fixed_split_csv=args.fixed_split_csv,
        copy_mode=args.copy_mode,
        seed=args.seed,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
