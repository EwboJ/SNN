"""
走廊导航 run 目录批量重命名
========================================
将旧命名格式 left1_bag1 / right2_bag3 / straight3_bag2
重命名为新格式   J1_left_r01 / J2_right_r03 / J3_straight_r02

用法:
  # 使用默认目录 (原地重命名)
  python scripts/rename_corridor_runs.py

  # 指定数据目录 (原地重命名)
  python scripts/rename_corridor_runs.py --data_dir ./data/corridor_all

  # 指定输出目录 (复制到新位置, 原目录不变)
  python scripts/rename_corridor_runs.py --data_dir ./data/corridor_all --out_dir ./data/corridor_renamed

  # 预览模式 (不执行, 仅打印)
  python scripts/rename_corridor_runs.py --dry_run
"""

import os
import re
import shutil
import argparse

# 默认数据目录
DEFAULT_DATA_DIR = './data/corridor_balanced'

# 新命名规则：J{junction}_{direction}_r{采集编号}
# 原命名格式：left1_bag1/right2_bag3/straight3_bag2
# direction: left/right/straight
# junction: 数字
# bag: 采集编号

def rename_runs(data_dir, out_dir=None, dry_run=False):
    """
    批量重命名 run 目录。

    Args:
        data_dir: 源数据目录
        out_dir:  输出目录。None 则原地重命名；指定则复制到新目录
        dry_run:  True 仅打印计划，不执行
    """
    if not os.path.isdir(data_dir):
        print(f'  ✗ 数据目录不存在: {data_dir}')
        return

    if out_dir and not dry_run:
        os.makedirs(out_dir, exist_ok=True)

    pattern = re.compile(r'^(left|right|straight)(\d+)_bag(\d+)$')
    renamed, skipped = 0, 0

    for name in sorted(os.listdir(data_dir)):
        old_path = os.path.join(data_dir, name)
        if not os.path.isdir(old_path):
            continue
        m = pattern.match(name)
        if not m:
            print(f'  Skip: {name}')
            skipped += 1
            continue
        direction, junction, bag = m.groups()
        new_name = f'J{junction}_{direction}_r{int(bag):02d}'

        target_dir = out_dir if out_dir else data_dir
        new_path = os.path.join(target_dir, new_name)

        if os.path.exists(new_path):
            print(f'  目标已存在: {new_name}, 跳过')
            skipped += 1
            continue

        if dry_run:
            print(f'  [dry_run] {name} -> {new_name}')
        elif out_dir:
            print(f'  {name} -> {new_name}  (复制)')
            shutil.copytree(old_path, new_path)
        else:
            print(f'  {name} -> {new_name}')
            os.rename(old_path, new_path)
        renamed += 1

    print(f'\n  完成: 重命名 {renamed} 个, 跳过 {skipped} 个')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='走廊导航 run 目录批量重命名',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='源数据目录')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='输出目录 (不指定则原地重命名)')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅预览，不执行')
    args = parser.parse_args()

    print(f'数据目录: {os.path.abspath(args.data_dir)}')
    if args.out_dir:
        print(f'输出目录: {os.path.abspath(args.out_dir)}')
    else:
        print(f'模式: 原地重命名')
    if args.dry_run:
        print(f'[dry_run 模式]')
    print()

    rename_runs(args.data_dir, out_dir=args.out_dir, dry_run=args.dry_run)

