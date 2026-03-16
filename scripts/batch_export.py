#!/usr/bin/env python3
"""
batch_export.py — 批量从 ROS2 bag 导出走廊数据集
==================================================
自动扫描 bag 目录下所有 bag，调用 corridor_export.run_export() 逐个导出。

使用:
  python batch_export.py --bag_dir ../bag_data --output_dir ./data/corridor_all
  python batch_export.py --bag_dir ../bag_data --output_dir ./data/corridor_all --config corridor_config.yaml
  python batch_export.py --bag_dir ../bag_data --output_dir ./data/corridor_all --split 0.8
  python batch_export.py --bag_dir ../bag_data --output_dir ./data/corridor_all --dry_run

输出结构 (--split 0.8 时):
  output_dir/
  ├── train/
  │   ├── left1_bag1/
  │   ├── left1_bag2/
  │   └── ...
  └── test/
      ├── left1_bag6/
      └── ...

输出结构 (不指定 --split 时):
  output_dir/
  ├── left1_bag1/
  ├── left1_bag2/
  └── ...
"""

import os
import sys
import argparse
import time
import re
from collections import defaultdict

from corridor_export import load_config, run_export


def find_bags(bag_dir):
    """
    扫描 bag_dir 下所有 ROS2 bag 目录（含 metadata.yaml 的文件夹）。
    返回按名称排序的 (bag_name, bag_path) 列表。
    """
    bags = []
    for entry in sorted(os.listdir(bag_dir)):
        bag_path = os.path.join(bag_dir, entry)
        if not os.path.isdir(bag_path):
            continue
        # ROS2 bag 目录应包含 metadata.yaml
        if os.path.exists(os.path.join(bag_path, "metadata.yaml")):
            bags.append((entry, bag_path))
    return bags


def group_bags(bags):
    """
    将 bag 按类别分组，用于按组拆分 train/test。
    分组规则：bag 名称去掉末尾的 _bag数字 后作为组名。
    例: left1_bag1, left1_bag2 → 组 "left1"
    """
    groups = defaultdict(list)
    for name, path in bags:
        # left1_bag3 → group="left1"
        match = re.match(r'^(.+?)_bag\d+.*$', name)
        group_key = match.group(1) if match else name
        groups[group_key].append((name, path))
    return dict(groups)


def split_train_test(groups, train_ratio):
    """
    按组内拆分 train/test。每组内最后 N 个 bag 做测试。
    保证每个类别（left1/left2/right1 等）都有训练和测试样本。
    """
    train_bags = []
    test_bags = []
    for group_key in sorted(groups.keys()):
        bag_list = groups[group_key]
        n = len(bag_list)
        n_train = max(1, int(n * train_ratio))
        # 确保至少有 1 个测试
        if n_train >= n and n > 1:
            n_train = n - 1
        train_bags.extend(bag_list[:n_train])
        test_bags.extend(bag_list[n_train:])
    return train_bags, test_bags


def main():
    parser = argparse.ArgumentParser(
        description="批量导出走廊 ROS2 bag 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--bag_dir", type=str, required=True,
                        help="包含所有 bag 文件夹的根目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="导出输出根目录")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径 (默认使用内置配置)")
    parser.add_argument("--split", type=float, default=0,
                        help="[已弃用] train/test 拆分比例 (如 0.8 = 80%% 训练)，"
                             "0 表示不拆分。推荐使用 split_corridor_runs.py 或 "
                             "corridor_dataset_pipeline.py 进行 train/val/test 划分")
    parser.add_argument("--odom_topic", type=str, default=None,
                        help="里程计话题名称 (默认 /odom_raw)")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印计划，不实际导出")
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已存在的输出目录（断点续导）")

    args = parser.parse_args()

    # 加载配置
    cfg_template = load_config(args.config)

    # 扫描所有 bag
    bags = find_bags(args.bag_dir)
    if not bags:
        print(f"[ERROR] 未在 {args.bag_dir} 下找到任何 ROS2 bag (需包含 metadata.yaml)")
        sys.exit(1)

    print(f"[INFO] 找到 {len(bags)} 个 bag:")
    for name, _ in bags:
        print(f"  {name}")

    # 分组
    groups = group_bags(bags)
    print(f"\n[INFO] 分为 {len(groups)} 个组:")
    for gk in sorted(groups.keys()):
        names = [n for n, _ in groups[gk]]
        print(f"  {gk}: {len(names)} bags → {', '.join(names)}")

    # 拆分 train/test
    do_split = args.split > 0
    if do_split:
        train_bags, test_bags = split_train_test(groups, args.split)
        tasks = [(name, path, "train") for name, path in train_bags] + \
                [(name, path, "test") for name, path in test_bags]
        print(f"\n[INFO] 拆分 train/test (ratio={args.split}):")
        print(f"  训练: {len(train_bags)} bags")
        print(f"  测试: {len(test_bags)} bags")
    else:
        tasks = [(name, path, "") for name, path in bags]

    # 计算输出路径
    export_plan = []
    for name, bag_path, subset in tasks:
        if subset:
            out_path = os.path.join(args.output_dir, subset, name)
        else:
            out_path = os.path.join(args.output_dir, name)
        export_plan.append((name, bag_path, out_path, subset))

    # 打印导出计划
    print(f"\n{'='*70}")
    print(f"  导出计划 ({len(export_plan)} 个 bag)")
    print(f"{'='*70}")
    for i, (name, bag_path, out_path, subset) in enumerate(export_plan, 1):
        tag = f"[{subset}]" if subset else ""
        skip = " (SKIP - exists)" if args.skip_existing and os.path.exists(out_path) else ""
        print(f"  {i:3d}. {tag:7s} {name:30s} → {out_path}{skip}")

    if args.dry_run:
        print(f"\n[DRY RUN] 以上为导出计划，未实际执行。去掉 --dry_run 开始导出。")
        return

    # 执行导出
    print(f"\n{'='*70}")
    print(f"  开始批量导出...")
    print(f"{'='*70}")

    success = 0
    failed = []
    skipped = 0
    t_start = time.time()

    for i, (name, bag_path, out_path, subset) in enumerate(export_plan, 1):
        tag = f"[{subset}]" if subset else ""
        print(f"\n{'─'*70}")
        print(f"  [{i}/{len(export_plan)}] {tag} {name}")
        print(f"{'─'*70}")

        if args.skip_existing and os.path.exists(out_path):
            print(f"  [SKIP] 输出目录已存在: {out_path}")
            skipped += 1
            continue

        try:
            import copy
            cfg = copy.deepcopy(cfg_template)
            if args.odom_topic:
                cfg["bag"]["odom_topic"] = args.odom_topic
            os.makedirs(out_path, exist_ok=True)
            run_export(bag_path, out_path, cfg)
            success += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append((name, str(e)))

    elapsed = time.time() - t_start

    # 汇总
    print(f"\n{'='*70}")
    print(f"  批量导出完成!")
    print(f"{'='*70}")
    print(f"  成功: {success}")
    print(f"  跳过: {skipped}")
    print(f"  失败: {len(failed)}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  输出: {os.path.abspath(args.output_dir)}")

    if failed:
        print(f"\n  失败列表:")
        for name, err in failed:
            print(f"    {name}: {err}")

    if do_split:
        print(f"\n  下一步训练:")
        print(f"    python train.py --dataset corridor "
              f"--corridor_root {args.output_dir} "
              f"--mode discrete --action_set 3 -T 8")


if __name__ == "__main__":
    main()
