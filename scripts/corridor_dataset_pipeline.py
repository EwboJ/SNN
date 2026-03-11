"""
走廊导航数据集统一处理管线
========================================
从 rosbag 到最终训练数据的一站式处理流程:

    export -> downsample -> split -> derive

用法:
  # 全流程
  python scripts/corridor_dataset_pipeline.py --mode all --bag_dir ./data/bags --force

  # 仅准备 (export + downsample + split)
  python scripts/corridor_dataset_pipeline.py --mode prepare --bag_dir ./data/bags

  # 从已有降采样数据开始 (仅 split + derive)
  python scripts/corridor_dataset_pipeline.py --mode all --skip_export --skip_downsample

  # 仅派生
  python scripts/corridor_dataset_pipeline.py --mode derive_only

  # 仅导出
  python scripts/corridor_dataset_pipeline.py --mode export_only --bag_dir ./data/bags

各阶段独立脚本仍可单独使用:
  - corridor_export.py / batch_export.py
  - scripts/downsample_corridor.py
  - scripts/split_corridor_runs.py
  - scripts/derive_stage1_datasets.py
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime


# ============================================================================
# 常量 & 默认路径
# ============================================================================
DEFAULT_BAG_DIR = './data/bags'
DEFAULT_EXPORT_ROOT = './data/corridor_all'
DEFAULT_DOWNSAMPLE_ROOT = './data/corridor_balanced'
DEFAULT_SPLIT_ROOT = './data/corridor'
DEFAULT_DERIVE_ROOT = './data/stage1'

STAGES = ['export', 'downsample', 'split', 'derive']

MODE_STAGES = {
    'export_only': ['export'],
    'prepare':     ['export', 'downsample', 'split'],
    'derive_only': ['derive'],
    'all':         ['export', 'downsample', 'split', 'derive'],
}


# ============================================================================
# 阶段函数
# ============================================================================

def stage_export(args):
    """阶段 1: 从 rosbag 导出原始帧数据"""
    print(f'\n{"=" * 72}')
    print(f'  阶段 1/4: EXPORT (rosbag -> 原始帧)')
    print(f'{"=" * 72}')

    if not args.bag_dir or not os.path.isdir(args.bag_dir):
        print(f'  ✗ bag_dir 不存在或未指定: {args.bag_dir}')
        print(f'  请用 --bag_dir 指定 rosbag 文件目录')
        return False

    # 调用 batch_export
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.batch_export import main as batch_export_main
    except ImportError:
        # 也尝试直接从项目根目录导入
        try:
            from batch_export import main as batch_export_main
        except ImportError:
            print('  ✗ 无法导入 batch_export.py')
            print('  请确保项目结构完整')
            return False

    # 构造 batch_export 的命令行参数
    export_argv = [
        '--bag_dir', args.bag_dir,
        '--output_dir', args.export_root,
    ]
    if hasattr(args, 'img_h') and args.img_h:
        export_argv.extend(['--img_h', str(args.img_h)])
    if hasattr(args, 'img_w') and args.img_w:
        export_argv.extend(['--img_w', str(args.img_w)])
    if args.force:
        export_argv.append('--force')

    print(f'  命令等效: python scripts/batch_export.py {" ".join(export_argv)}')

    old_argv = sys.argv
    sys.argv = ['batch_export.py'] + export_argv
    try:
        batch_export_main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv

    print(f'\n  ✓ Export 完成 -> {os.path.abspath(args.export_root)}')
    return True


def stage_downsample(args):
    """阶段 2: 智能降采样"""
    print(f'\n{"=" * 72}')
    print(f'  阶段 2/4: DOWNSAMPLE (智能降采样)')
    print(f'{"=" * 72}')

    try:
        from scripts.downsample_corridor import run_downsample
    except ImportError:
        # 如果从项目根运行
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from downsample_corridor import run_downsample

    try:
        run_downsample(
            src_root=args.export_root,
            dst_root=args.downsample_root,
            context_frames=args.context_frames,
            stride=args.stride,
            copy_mode=args.copy_mode,
            exclude=args.exclude,
            valid_only=True,
            force=args.force,
        )
        print(f'\n  ✓ Downsample 完成 -> {os.path.abspath(args.downsample_root)}')
        return True
    except Exception as e:
        print(f'  ✗ Downsample 失败: {e}')
        return False


def stage_split(args):
    """阶段 3: train/val/test 划分"""
    print(f'\n{"=" * 72}')
    print(f'  阶段 3/4: SPLIT (train/val/test 划分)')
    print(f'{"=" * 72}')

    try:
        from scripts.split_corridor_runs import run_split
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from split_corridor_runs import run_split

    try:
        run_split(
            src_root=args.downsample_root,
            dst_root=args.split_root,
            split_mode=args.split_mode,
            group_by=args.group_by,
            train_per_group=args.train_per_group,
            val_per_group=args.val_per_group,
            test_per_group=args.test_per_group,
            copy_mode=args.copy_mode,
            seed=args.seed,
            force=args.force,
        )
        print(f'\n  ✓ Split 完成 -> {os.path.abspath(args.split_root)}')
        return True
    except Exception as e:
        print(f'  ✗ Split 失败: {e}')
        return False


def stage_derive(args):
    """阶段 4: 派生任务数据集"""
    print(f'\n{"=" * 72}')
    print(f'  阶段 4/4: DERIVE (派生任务数据集)')
    print(f'{"=" * 72}')

    try:
        from scripts.derive_stage1_datasets import main as derive_main
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from derive_stage1_datasets import main as derive_main

    # 构造 derive 的命令行参数
    derive_argv = [
        '--src_root', args.split_root,
        '--dst_root', args.derive_root,
        '--task', args.derive_task,
        '--seed', str(args.seed),
    ]
    if args.force:
        derive_argv.append('--force')

    # 传入自定义窗口参数
    if hasattr(args, 'pre_turn_ms') and args.pre_turn_ms is not None:
        derive_argv.extend(['--pre_turn_ms', str(args.pre_turn_ms)])
    if hasattr(args, 'recover_ms') and args.recover_ms is not None:
        derive_argv.extend(['--recover_ms', str(args.recover_ms)])

    old_argv = sys.argv
    sys.argv = ['derive_stage1_datasets.py'] + derive_argv
    try:
        derive_main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv

    print(f'\n  ✓ Derive 完成 -> {os.path.abspath(args.derive_root)}')
    return True


# ============================================================================
# 管线编排
# ============================================================================

STAGE_FUNCS = {
    'export': stage_export,
    'downsample': stage_downsample,
    'split': stage_split,
    'derive': stage_derive,
}


def main():
    parser = argparse.ArgumentParser(
        description='走廊导航数据集统一处理管线 '
                    '(export -> downsample -> split -> derive)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- 模式 ----
    parser.add_argument('--mode', type=str, default='all',
                        choices=list(MODE_STAGES.keys()),
                        help='管线模式')

    # ---- 跳过 ----
    parser.add_argument('--skip_export', action='store_true',
                        help='跳过 export 阶段')
    parser.add_argument('--skip_downsample', action='store_true',
                        help='跳过 downsample 阶段')
    parser.add_argument('--skip_split', action='store_true',
                        help='跳过 split 阶段')
    parser.add_argument('--skip_derive', action='store_true',
                        help='跳过 derive 阶段')

    # ---- 路径 ----
    parser.add_argument('--bag_dir', type=str, default=DEFAULT_BAG_DIR,
                        help='rosbag 文件目录 (export 阶段用)')
    parser.add_argument('--export_root', type=str, default=DEFAULT_EXPORT_ROOT,
                        help='export 输出 / downsample 输入')
    parser.add_argument('--downsample_root', type=str,
                        default=DEFAULT_DOWNSAMPLE_ROOT,
                        help='downsample 输出 / split 输入')
    parser.add_argument('--split_root', type=str, default=DEFAULT_SPLIT_ROOT,
                        help='split 输出 / derive 输入')
    parser.add_argument('--derive_root', type=str, default=DEFAULT_DERIVE_ROOT,
                        help='derive 输出')

    # ---- Export 参数 ----
    parser.add_argument('--img_h', type=int, default=None,
                        help='导出图片高度 (None=原始)')
    parser.add_argument('--img_w', type=int, default=None,
                        help='导出图片宽度 (None=原始)')

    # ---- Downsample 参数 ----
    parser.add_argument('--context_frames', type=int, default=15,
                        help='转弯上下文半径')
    parser.add_argument('--stride', type=int, default=3,
                        help='远 Forward 帧抽样步长')
    parser.add_argument('--exclude', nargs='*', default=[],
                        help='排除的 run 名称')

    # ---- Split 参数 ----
    parser.add_argument('--split_mode', type=str, default='exact',
                        choices=['exact', 'ratio'],
                        help='划分模式')
    parser.add_argument('--group_by', type=str,
                        default='junction_id,turn_dir',
                        help='分组键')
    parser.add_argument('--train_per_group', type=int, default=5)
    parser.add_argument('--val_per_group', type=int, default=1)
    parser.add_argument('--test_per_group', type=int, default=1)

    # ---- Derive 参数 ----
    parser.add_argument('--derive_task', type=str, default='all',
                        choices=['action3_balanced', 'junction_lr',
                                 'stage4', 'all'],
                        help='派生任务')
    parser.add_argument('--pre_turn_ms', type=float, default=None,
                        help='turn_on 前窗口 (ms)')
    parser.add_argument('--recover_ms', type=float, default=None,
                        help='turn_off 后 recover 窗口 (ms)')

    # ---- 通用 ----
    parser.add_argument('--copy_mode', type=str, default='copy',
                        choices=['copy', 'symlink'],
                        help='文件复制方式')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true',
                        help='覆盖已有输出')
    parser.add_argument('--dry_run', action='store_true',
                        help='仅预览不执行')

    args = parser.parse_args()

    # ======================== 确定执行阶段 ========================
    stages = list(MODE_STAGES[args.mode])

    skip_map = {
        'export': args.skip_export,
        'downsample': args.skip_downsample,
        'split': args.skip_split,
        'derive': args.skip_derive,
    }
    stages = [s for s in stages if not skip_map.get(s, False)]

    if not stages:
        print('没有需要执行的阶段 (全部被跳过)')
        return

    # ======================== Banner ========================
    print('=' * 72)
    print('  走廊导航数据集统一处理管线')
    print('=' * 72)
    print(f'  模式:      {args.mode}')
    print(f'  执行阶段:  {" -> ".join(stages)}')
    print(f'  路径:')
    if 'export' in stages:
        print(f'    bags      : {os.path.abspath(args.bag_dir)}')
        print(f'    export    : {os.path.abspath(args.export_root)}')
    if 'downsample' in stages:
        print(f'    downsample: {os.path.abspath(args.downsample_root)}')
    if 'split' in stages:
        print(f'    split     : {os.path.abspath(args.split_root)}')
    if 'derive' in stages:
        print(f'    derive    : {os.path.abspath(args.derive_root)}')
    print(f'  Force:     {args.force}')
    print(f'  Seed:      {args.seed}')
    print('=' * 72)

    # ======================== 执行 ========================
    t0 = time.time()
    results = {}

    for stage in stages:
        ts = time.time()
        print(f'\n{"#" * 72}')
        print(f'  开始阶段: {stage.upper()}')
        print(f'{"#" * 72}')

        func = STAGE_FUNCS[stage]
        ok = func(args)
        elapsed = time.time() - ts

        results[stage] = {
            'success': ok,
            'elapsed_s': round(elapsed, 1),
        }

        if not ok:
            print(f'\n  ✗ 阶段 {stage} 失败 (耗时 {elapsed:.1f}s)')
            print(f'  后续阶段已跳过')
            break
        else:
            print(f'\n  ✓ 阶段 {stage} 完成 (耗时 {elapsed:.1f}s)')

    total_elapsed = time.time() - t0

    # ======================== 总结 ========================
    print(f'\n{"=" * 72}')
    print(f'  管线执行总结')
    print(f'{"=" * 72}')
    for stage, info in results.items():
        status = '✓' if info['success'] else '✗'
        print(f'  [{status}] {stage:12s}  {info["elapsed_s"]:6.1f}s')
    print(f'  {"─" * 30}')
    print(f'  总耗时: {total_elapsed:.1f}s')

    # 保存管线日志
    pipeline_log = {
        'timestamp': datetime.now().isoformat(),
        'mode': args.mode,
        'stages_executed': list(results.keys()),
        'results': results,
        'total_elapsed_s': round(total_elapsed, 1),
        'paths': {
            'export_root': args.export_root,
            'downsample_root': args.downsample_root,
            'split_root': args.split_root,
            'derive_root': args.derive_root,
        },
    }

    # 保存到 derive_root 或 split_root
    log_dir = args.derive_root if 'derive' in results else args.split_root
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'pipeline_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(pipeline_log, f, indent=2, ensure_ascii=False)
    print(f'\n  管线日志: {log_path}')

    all_ok = all(r['success'] for r in results.values())
    if all_ok:
        print(f'\n  ✓ 管线全部完成!')
    else:
        print(f'\n  ✗ 管线存在失败阶段，请检查日志')
    print('=' * 72)


if __name__ == '__main__':
    main()
