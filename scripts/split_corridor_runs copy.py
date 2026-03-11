"""
走廊数据集 train/test 划分脚本
=====================================
将 corridor_export.py 导出的多个 run 目录按 run 为单位划分到 train/test。

用法:
    python split_corridor_runs.py --src_root ./data --dst_root ./data/corridor --test_ratio 0.2

输出:
    <dst_root>/
      train/left_run1/  train/right_run2/  ...
      test/left_run3/   test/right_run5/   ...
"""

import os
import sys
import json
import shutil
import random
import argparse
import csv
from pathlib import Path


def find_valid_runs(src_root: str) -> list:
    """
    扫描 src_root 下所有有效的 run 目录。

    有效条件: 包含 images/ 子目录 + labels.csv 文件。

    Returns:
        [(run_name, run_path, frame_count), ...]
    """
    runs = []
    for name in sorted(os.listdir(src_root)):
        rp = os.path.join(src_root, name)
        if not os.path.isdir(rp):
            continue
        img_dir = os.path.join(rp, 'images')
        lbl_csv = os.path.join(rp, 'labels.csv')
        if not os.path.isdir(img_dir) or not os.path.isfile(lbl_csv):
            print(f"  [跳过] {name}: 缺少 images/ 或 labels.csv")
            continue

        # 统计帧数
        frame_count = 0
        meta_path = os.path.join(rp, 'meta.json')
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                frame_count = meta.get('total_frames', 0)
            except Exception:
                pass

        if frame_count == 0:
            # fallback: 数 labels.csv 行数
            try:
                with open(lbl_csv, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # skip header
                    frame_count = sum(1 for _ in reader)
            except Exception:
                pass

        runs.append((name, rp, frame_count))

    return runs


def get_action_dist(run_path: str) -> dict:
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
    lbl_csv = os.path.join(run_path, 'labels.csv')
    dist = {}
    try:
        with open(lbl_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                act = row.get('action_name', 'unknown')
                dist[act] = dist.get(act, 0) + 1
    except Exception:
        pass
    return dist


def copy_run(src: str, dst: str, mode: str):
    """复制或创建符号链接"""
    if mode == 'symlink':
        src_abs = os.path.abspath(src)
        os.symlink(src_abs, dst, target_is_directory=True)
    else:
        shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description='将走廊 run 目录划分为 train/test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_root', type=str, default='./data',
                        help='原始 run 目录的根路径')
    parser.add_argument('--dst_root', type=str, default='./data/corridor',
                        help='输出目录 (会创建 train/ 和 test/ 子目录)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='测试集 run 比例 (0.0~1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (确保可复现)')
    parser.add_argument('--copy_mode', type=str, default='copy',
                        choices=['copy', 'symlink'],
                        help='复制方式: copy=完整复制, symlink=符号链接')
    parser.add_argument('--force', action='store_true',
                        help='如果目标目录已存在，不提示直接覆盖')

    args = parser.parse_args()

    print('=' * 70)
    print('  走廊数据集 Train/Test 划分')
    print('=' * 70)
    print(f'  源目录:      {os.path.abspath(args.src_root)}')
    print(f'  目标目录:    {os.path.abspath(args.dst_root)}')
    print(f'  测试比例:    {args.test_ratio}')
    print(f'  随机种子:    {args.seed}')
    print(f'  复制方式:    {args.copy_mode}')
    print('=' * 70)

    # ---- 扫描有效 run ----
    print(f'\n[1/4] 扫描 run 目录...')
    runs = find_valid_runs(args.src_root)

    if len(runs) == 0:
        print(f"  ✗ 在 {args.src_root} 下未找到任何有效 run 目录!")
        print(f"  有效 run 需包含 images/ 子目录和 labels.csv 文件")
        sys.exit(1)

    print(f'  找到 {len(runs)} 个有效 run:')
    total_frames = 0
    for name, path, fc in runs:
        ad = get_action_dist(path)
        ad_str = ', '.join(f'{k}:{v}' for k, v in ad.items())
        print(f'    {name:20s}  {fc:5d} 帧  [{ad_str}]')
        total_frames += fc
    print(f'  总帧数: {total_frames}')

    # ---- 划分 ----
    print(f'\n[2/4] 按 run 划分 train/test...')
    random.seed(args.seed)
    indices = list(range(len(runs)))
    random.shuffle(indices)

    n_test = max(1, round(len(runs) * args.test_ratio))
    n_train = len(runs) - n_test

    # 确保至少各有 1 个
    if n_train < 1:
        n_train = 1
        n_test = len(runs) - 1
    if n_test < 1:
        n_test = 1
        n_train = len(runs) - 1

    test_indices = set(indices[:n_test])
    train_runs = [(runs[i]) for i in range(len(runs)) if i not in test_indices]
    test_runs = [(runs[i]) for i in range(len(runs)) if i in test_indices]

    train_frames = sum(fc for _, _, fc in train_runs)
    test_frames = sum(fc for _, _, fc in test_runs)

    print(f'  Train: {len(train_runs)} runs, {train_frames} 帧')
    for name, _, fc in train_runs:
        print(f'    → {name} ({fc} 帧)')
    print(f'  Test:  {len(test_runs)} runs, {test_frames} 帧')
    for name, _, fc in test_runs:
        print(f'    → {name} ({fc} 帧)')

    # ---- 检查目标目录 ----
    train_dir = os.path.join(args.dst_root, 'train')
    test_dir = os.path.join(args.dst_root, 'test')

    if os.path.exists(train_dir) or os.path.exists(test_dir):
        if not args.force:
            print(f'\n  ⚠ 目标目录已存在:')
            if os.path.exists(train_dir):
                print(f'    {train_dir}')
            if os.path.exists(test_dir):
                print(f'    {test_dir}')
            ans = input('  是否覆盖? [y/N] ').strip().lower()
            if ans not in ('y', 'yes'):
                print('  已取消。')
                sys.exit(0)

        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    # ---- 执行复制 ----
    print(f'\n[3/4] 复制 run 到目标目录 (mode={args.copy_mode})...')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for name, path, fc in train_runs:
        dst = os.path.join(train_dir, name)
        print(f'  [train] {name} → {dst}')
        copy_run(path, dst, args.copy_mode)

    for name, path, fc in test_runs:
        dst = os.path.join(test_dir, name)
        print(f'  [test]  {name} → {dst}')
        copy_run(path, dst, args.copy_mode)

    # ---- 统计汇总 ----
    print(f'\n[4/4] 划分完成!')
    print('=' * 70)
    print(f'  目标目录:  {os.path.abspath(args.dst_root)}')
    print(f'  Train:     {len(train_runs):3d} runs, {train_frames:6d} 帧')
    print(f'  Test:      {len(test_runs):3d} runs, {test_frames:6d} 帧')
    print(f'  总计:      {len(runs):3d} runs, {total_frames:6d} 帧')
    print(f'  实际比例:  train={100*len(train_runs)/len(runs):.0f}% / '
          f'test={100*len(test_runs)/len(runs):.0f}%')
    print('=' * 70)

    # 按动作统计 train/test
    print(f'\n  Train 动作分布:')
    train_ad = {}
    for name, path, _ in train_runs:
        for k, v in get_action_dist(path).items():
            train_ad[k] = train_ad.get(k, 0) + v
    for k, v in sorted(train_ad.items()):
        print(f'    {k:10s}: {v}')

    print(f'  Test 动作分布:')
    test_ad = {}
    for name, path, _ in test_runs:
        for k, v in get_action_dist(path).items():
            test_ad[k] = test_ad.get(k, 0) + v
    for k, v in sorted(test_ad.items()):
        print(f'    {k:10s}: {v}')

    print(f'\n  下一步训练命令:')
    print(f'    python train.py --dataset corridor '
          f'--corridor_root {args.dst_root} '
          f'--mode discrete --action_set 3 -T 8')


if __name__ == '__main__':
    main()
