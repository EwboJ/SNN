"""
走廊数据集智能降采样脚本
============================
保留全部 Left/Right/Stop 帧，对 Forward 帧分两类处理：
  a) 转弯上下文区（距最近 Left/Right ≤ context_frames）：全部保留
  b) 远离转弯的长直行段：按 stride 步长抽样

输出新的平衡版数据集到指定目录，保持目录结构。
仅复制/软链接保留的图片，不复制被丢弃的图片。

用法:
    python scripts/downsample_corridor.py --src_root ./data/corridor_all --dst_root ./data/corridor_balanced --context_frames 15 --stride 3 --copy_mode copy --exclude left4_bag2

"""

import os
import sys
import csv
import json
import shutil
import argparse
from collections import OrderedDict, defaultdict


def load_labels(csv_path):
    """读取 labels.csv，返回 [{...}, ...] 列表"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['action_id'] = int(row['action_id'])
            row['valid'] = int(row.get('valid', '1'))
            rows.append(row)
    return rows


def compute_turn_distance(rows):
    """
    为每一帧计算到最近 Left/Right 帧的距离（帧数）。

    Returns:
        distances: list[int]  每帧到最近转弯帧的距离
    """
    n = len(rows)
    distances = [n] * n  # 初始化为最大距离

    # 找到所有转弯帧的索引
    turn_indices = []
    for i, row in enumerate(rows):
        if row['action_name'] in ('Left', 'Right') and row['valid'] == 1:
            turn_indices.append(i)

    if not turn_indices:
        return distances

    # 向前扫描：每帧到之前最近转弯帧的距离
    for i in range(n):
        for ti in turn_indices:
            d = abs(i - ti)
            if d < distances[i]:
                distances[i] = d

    return distances


def downsample_run(rows, context_frames, stride, valid_only=True):
    """
    对单个 run 的帧做智能降采样。

    规则:
      1) Left/Right 帧：全部保留
      2) Stop 帧：全部保留（数量本来就很少）
      3) Forward 帧，距「最近转弯帧 ≤ context_frames」：全部保留（转弯上下文）
      4) Forward 帧，距「最近转弯帧 > context_frames」：每 stride 帧取 1 帧
      5) invalid 帧（valid=0）：丢弃
      6) 额外保证：每个 run 的第 1 帧和最后 1 帧始终保留

    Args:
        rows: labels.csv 解析后的列表
        context_frames: 转弯上下文半径（帧数）
        stride: 远离转弯的 Forward 帧的抽样步长（每 stride 帧取 1）
        valid_only: 是否丢弃 invalid 帧

    Returns:
        keep_mask: list[bool]  是否保留该帧
        reasons: list[str]     保留/丢弃原因
    """
    n = len(rows)
    keep_mask = [False] * n
    reasons = [''] * n

    # 计算到最近转弯帧的距离
    distances = compute_turn_distance(rows)

    # 远离转弯的 Forward 帧：用计数器做步长采样
    far_forward_counter = 0

    for i, row in enumerate(rows):
        action = row['action_name']
        valid = row['valid']

        # 丢弃无效帧
        if valid_only and valid == 0:
            reasons[i] = 'invalid'
            continue

        # 规则1: Left/Right → 全部保留
        if action in ('Left', 'Right'):
            keep_mask[i] = True
            reasons[i] = 'turn'
            far_forward_counter = 0
            continue

        # 规则2: Stop → 全部保留
        if action == 'Stop':
            keep_mask[i] = True
            reasons[i] = 'stop'
            continue

        # 规则3: Forward + 转弯上下文 → 保留
        if action == 'Forward' and distances[i] <= context_frames:
            keep_mask[i] = True
            reasons[i] = 'context'
            far_forward_counter = 0
            continue

        # 规则4: Forward + 远离转弯 → 按步长采样
        if action == 'Forward':
            if far_forward_counter % stride == 0:
                keep_mask[i] = True
                reasons[i] = 'sampled'
            else:
                reasons[i] = 'dropped'
            far_forward_counter += 1
            continue

        # 其他未知动作 → 保留
        keep_mask[i] = True
        reasons[i] = 'other'

    # 规则6: 首尾帧始终保留
    if n > 0:
        if rows[0]['valid'] == 1 or not valid_only:
            keep_mask[0] = True
            reasons[0] = 'first'
        if rows[-1]['valid'] == 1 or not valid_only:
            keep_mask[-1] = True
            reasons[-1] = 'last'

    return keep_mask, reasons


def count_actions(rows, mask=None):
    """统计动作分布 (可选只统计 mask=True 的帧)"""
    dist = defaultdict(int)
    for i, row in enumerate(rows):
        if mask is not None and not mask[i]:
            continue
        if row['valid'] == 0:
            continue
        dist[row['action_name']] += 1
    return dict(dist)


def process_run(run_name, src_run_dir, dst_run_dir, context_frames, stride,
                copy_mode, valid_only):
    """
    处理单个 run：降采样 + 复制保留的帧。

    Returns:
        stats dict
    """
    lbl_path = os.path.join(src_run_dir, 'labels.csv')
    img_dir = os.path.join(src_run_dir, 'images')

    if not os.path.isfile(lbl_path) or not os.path.isdir(img_dir):
        print(f"  [跳过] {run_name}: 缺少 labels.csv 或 images/")
        return None

    rows = load_labels(lbl_path)
    keep_mask, reasons = downsample_run(rows, context_frames, stride,
                                        valid_only)

    # 统计
    before_dist = count_actions(rows)
    after_dist = count_actions(rows, keep_mask)
    n_before = sum(before_dist.values())
    n_after = sum(after_dist.values())

    # 创建目标目录
    dst_img_dir = os.path.join(dst_run_dir, 'images')
    os.makedirs(dst_img_dir, exist_ok=True)

    # 写新的 labels.csv (只保留 kept 帧)
    kept_rows = []
    for i, row in enumerate(rows):
        if keep_mask[i]:
            kept_rows.append(row)

    fieldnames = ['image_name', 'action_id', 'action_name', 'timestamp_ns',
                  'linear_x', 'angular_z', 'time_diff_ms', 'valid']
    with open(os.path.join(dst_run_dir, 'labels.csv'), 'w',
              newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        for row in kept_rows:
            writer.writerow(row)

    # 复制/链接保留的图片
    for row in kept_rows:
        img_name = row['image_name']
        src_img = os.path.join(img_dir, img_name)
        dst_img = os.path.join(dst_img_dir, img_name)
        if os.path.isfile(src_img):
            if copy_mode == 'symlink':
                src_abs = os.path.abspath(src_img)
                if not os.path.exists(dst_img):
                    os.symlink(src_abs, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

    # 复制 meta.json (如果存在)
    meta_src = os.path.join(src_run_dir, 'meta.json')
    if os.path.isfile(meta_src):
        # 更新 meta 中的帧数
        try:
            with open(meta_src, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            meta['original_total_frames'] = meta.get('total_frames', 0)
            meta['total_frames'] = n_after
            meta['valid_frames'] = n_after
            meta['action_distribution'] = after_dist
            meta['downsample_config'] = {
                'context_frames': context_frames,
                'stride': stride,
                'dropped_frames': n_before - n_after,
            }
            with open(os.path.join(dst_run_dir, 'meta.json'), 'w',
                       encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception:
            shutil.copy2(meta_src, os.path.join(dst_run_dir, 'meta.json'))

    # 统计 reason 分布
    reason_counts = defaultdict(int)
    for r in reasons:
        if r:
            reason_counts[r] += 1

    return {
        'run': run_name,
        'before': before_dist,
        'after': after_dist,
        'n_before': n_before,
        'n_after': n_after,
        'drop_rate': 1.0 - n_after / max(n_before, 1),
        'reasons': dict(reason_counts),
    }


def main():
    parser = argparse.ArgumentParser(
        description='走廊数据集智能降采样',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_root', type=str, default='./data/corridor_all',
                        help='原始数据根目录')
    parser.add_argument('--dst_root', type=str,
                        default='./data/corridor_balanced',
                        help='输出目录')
    parser.add_argument('--context_frames', type=int, default=15,
                        help='转弯上下文半径: 距最近 Left/Right ≤ N帧的 '
                             'Forward 全保留 (30FPS 下 15帧≈0.5秒)')
    parser.add_argument('--stride', type=int, default=3,
                        help='远离转弯的 Forward 帧抽样步长 '
                             '(3=每3帧取1, 即10FPS)')
    parser.add_argument('--copy_mode', type=str, default='copy',
                        choices=['copy', 'symlink'],
                        help='图片复制方式')
    parser.add_argument('--exclude', nargs='*', default=[''],
                        help='排除的 run 名称列表')
    parser.add_argument('--valid_only', type=lambda x:
                        str(x).lower() in ('true', '1', 'yes'),
                        default=True, help='是否丢弃 valid=0 帧')
    parser.add_argument('--force', action='store_true',
                        help='覆盖已有输出目录')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅打印统计，不实际复制文件')

    args = parser.parse_args()

    # 检查源目录
    if not os.path.isdir(args.src_root):
        print(f"错误: 源目录不存在: {args.src_root}")
        sys.exit(1)

    # 检查输出目录
    if os.path.exists(args.dst_root) and not args.dry_run:
        if not args.force:
            ans = input(f"输出目录已存在: {args.dst_root}\n是否覆盖? [y/N] ")
            if ans.strip().lower() not in ('y', 'yes'):
                print("已取消。")
                sys.exit(0)
        shutil.rmtree(args.dst_root)

    exclude_set = set(args.exclude) if args.exclude else set()

    print('=' * 75)
    print('  走廊数据智能降采样')
    print('=' * 75)
    print(f'  源目录:           {os.path.abspath(args.src_root)}')
    print(f'  输出目录:         {os.path.abspath(args.dst_root)}')
    print(f'  转弯上下文半径:   {args.context_frames} 帧 '
          f'(≈{args.context_frames/30:.1f}s @30FPS)')
    print(f'  远直行抽样步长:   每 {args.stride} 帧取 1')
    print(f'  排除 run:         {exclude_set or "无"}')
    print(f'  模式:             {"DRY RUN (不复制)" if args.dry_run else args.copy_mode}')
    print('=' * 75)

    # 处理每个 run
    all_stats = []
    total_before = defaultdict(int)
    total_after = defaultdict(int)

    runs = sorted([d for d in os.listdir(args.src_root)
                   if os.path.isdir(os.path.join(args.src_root, d))])

    for run_name in runs:
        if run_name in exclude_set:
            print(f"  [排除] {run_name}")
            continue

        src_dir = os.path.join(args.src_root, run_name)

        if args.dry_run:
            # Dry run: 只统计不复制
            lbl_path = os.path.join(src_dir, 'labels.csv')
            if not os.path.isfile(lbl_path):
                continue
            rows = load_labels(lbl_path)
            keep_mask, reasons = downsample_run(
                rows, args.context_frames, args.stride, args.valid_only)
            before_dist = count_actions(rows)
            after_dist = count_actions(rows, keep_mask)
            n_before = sum(before_dist.values())
            n_after = sum(after_dist.values())
            reason_counts = defaultdict(int)
            for r in reasons:
                if r:
                    reason_counts[r] += 1
            stats = {
                'run': run_name, 'before': before_dist,
                'after': after_dist, 'n_before': n_before,
                'n_after': n_after,
                'drop_rate': 1.0 - n_after / max(n_before, 1),
                'reasons': dict(reason_counts),
            }
        else:
            dst_dir = os.path.join(args.dst_root, run_name)
            stats = process_run(run_name, src_dir, dst_dir,
                                args.context_frames, args.stride,
                                args.copy_mode, args.valid_only)

        if stats:
            all_stats.append(stats)
            for k, v in stats['before'].items():
                total_before[k] += v
            for k, v in stats['after'].items():
                total_after[k] += v

    # ======================== 打印报告 ========================
    print(f'\n{"=" * 75}')
    print(f'  降采样结果')
    print(f'{"=" * 75}')

    # 每个 run 的详情
    print(f'\n{"Run":22s} {"前":>5s} {"后":>5s} {"降%":>5s}  '
          f'{"Fwd前":>5s}→{"Fwd后":>5s}  {"L前":>4s}→{"L后":>4s}  '
          f'{"R前":>4s}→{"R后":>4s}  {"Stp前":>5s}→{"Stp后":>5s}')
    print('-' * 100)

    for s in all_stats:
        b, a = s['before'], s['after']
        print(f"{s['run']:22s} {s['n_before']:5d} {s['n_after']:5d} "
              f"{s['drop_rate']:5.0%}  "
              f"{b.get('Forward',0):5d}→{a.get('Forward',0):5d}  "
              f"{b.get('Left',0):4d}→{a.get('Left',0):4d}  "
              f"{b.get('Right',0):4d}→{a.get('Right',0):4d}  "
              f"{b.get('Stop',0):5d}→{a.get('Stop',0):5d}")

    # 汇总
    n_total_before = sum(total_before.values())
    n_total_after = sum(total_after.values())

    print('-' * 100)
    print(f'{"TOTAL":22s} {n_total_before:5d} {n_total_after:5d} '
          f'{1.0-n_total_after/max(n_total_before,1):5.0%}  '
          f'{total_before.get("Forward",0):5d}→{total_after.get("Forward",0):5d}  '
          f'{total_before.get("Left",0):4d}→{total_after.get("Left",0):4d}  '
          f'{total_before.get("Right",0):4d}→{total_after.get("Right",0):4d}  '
          f'{total_before.get("Stop",0):5d}→{total_after.get("Stop",0):5d}')

    # 3类统计对比
    before_straight = total_before.get('Forward', 0) + total_before.get('Stop', 0)
    before_left = total_before.get('Left', 0)
    before_right = total_before.get('Right', 0)
    after_straight = total_after.get('Forward', 0) + total_after.get('Stop', 0)
    after_left = total_after.get('Left', 0)
    after_right = total_after.get('Right', 0)

    print(f'\n  ===== 3类分布对比 =====')
    print(f'  {"类别":10s} {"降采前":>8s} {"占比":>6s}  →  {"降采后":>8s} {"占比":>6s}')
    print(f'  {"─"*50}')
    for name, bv, av in [
        ('Left', before_left, after_left),
        ('Straight', before_straight, after_straight),
        ('Right', before_right, after_right),
    ]:
        bp = 100 * bv / max(n_total_before, 1)
        ap = 100 * av / max(n_total_after, 1)
        print(f'  {name:10s} {bv:8d} {bp:5.1f}%  →  {av:8d} {ap:5.1f}%')
    print(f'  {"Total":10s} {n_total_before:8d}        →  {n_total_after:8d}')

    # 失衡比率
    min_class = min(after_left, after_right) if after_left > 0 and after_right > 0 else 1
    imbalance = after_straight / max(min_class, 1)
    print(f'\n  Straight/min(L,R) 失衡比: {imbalance:.2f}× '
          f'(目标 ≤1.5×, {"✅ 达标" if imbalance <= 1.5 else "⚠️ 仍偏高"})')

    # 保留原因分布
    total_reasons = defaultdict(int)
    for s in all_stats:
        for k, v in s['reasons'].items():
            total_reasons[k] += v
    print(f'\n  ===== 保留/丢弃原因 =====')
    for r in ['turn', 'stop', 'context', 'sampled', 'first', 'last',
              'dropped', 'invalid', 'other']:
        if total_reasons.get(r, 0) > 0:
            print(f'  {r:12s}: {total_reasons[r]:6d}')

    # 保存汇总到 JSON
    if not args.dry_run:
        summary = {
            'config': {
                'src_root': args.src_root,
                'dst_root': args.dst_root,
                'context_frames': args.context_frames,
                'stride': args.stride,
                'exclude': list(exclude_set),
            },
            'before': dict(total_before),
            'after': dict(total_after),
            'n_before': n_total_before,
            'n_after': n_total_after,
            'drop_rate': 1.0 - n_total_after / max(n_total_before, 1),
            'runs': [{k: v for k, v in s.items()} for s in all_stats],
        }
        os.makedirs(args.dst_root, exist_ok=True)
        with open(os.path.join(args.dst_root, 'downsample_summary.json'),
                  'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f'\n  汇总已保存: {args.dst_root}/downsample_summary.json')

    print(f'\n{"=" * 75}')
    if args.dry_run:
        print(f'  [DRY RUN] 以上为预估结果，未实际复制文件。')
        print(f'  去掉 --dry_run 参数后重新运行即可执行。')
    else:
        print(f'  降采样完成! 输出: {os.path.abspath(args.dst_root)}')
        print(f'  处理: {len(all_stats)} 个 run, '
              f'{n_total_before}→{n_total_after} 帧 '
              f'(保留 {100*n_total_after/max(n_total_before,1):.0f}%)')
    print(f'{"=" * 75}')


if __name__ == '__main__':
    main()
