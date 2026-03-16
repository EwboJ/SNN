"""
走廊导航数据集统一处理管线
========================================
支持不同数据类型 (--task_type) 选择不同处理流程:

  junction:       export -> downsample -> split -> derive (stage1)
  straight_keep:  export -> split -> derive_straight_keep
  loop:           export -> split [-> extract_windows]
  generic:        export -> downsample -> split -> derive (同 junction)

用法:
  # junction 全流程
  python scripts/corridor_dataset_pipeline.py --task_type junction --mode all --bag_dir ./data/bags --force

  # straight_keep (默认跳过 downsample)
  python scripts/corridor_dataset_pipeline.py --task_type straight_keep --mode all --bag_dir ./data/bags --force

  # loop (导出+划分+提取窗口)
  python scripts/corridor_dataset_pipeline.py --task_type loop --mode all --loop_extract_windows --force

  # 兼容旧用法 (generic = junction)
  python scripts/corridor_dataset_pipeline.py --mode all --force

各阶段独立脚本仍可单独使用:
  - corridor_export.py / batch_export.py
  - scripts/downsample_corridor.py
  - scripts/split_corridor_runs.py
  - scripts/derive_stage1_datasets.py
  - scripts/derive_straight_keep_dataset.py
  - scripts/extract_loop_windows.py
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime


# ============================================================================
# 常量 & 默认路径 (按 task_type 区分)
# ============================================================================

TASK_TYPE_DEFAULTS = {
    'junction': {
        'export_root':     './data/junction_all',
        'downsample_root': './data/junction_balanced',
        'split_root':      './data/junction',
        'derive_root':     './data/stage1',
        'stages':          ['export', 'downsample', 'split', 'derive'],
        'auto_skip':       [],
    },
    'straight_keep': {
        'export_root':     './data/straight_keep_all',
        'downsample_root': '',   # 不使用
        'split_root':      './data/straight_keep',
        'derive_root':     './data/straight_keep/straight_keep_reg_v1',
        'stages':          ['export', 'split', 'derive_straight_keep'],
        'auto_skip':       ['downsample'],
    },
    'loop': {
        'export_root':     './data/loop_eval_raw',
        'downsample_root': '',   # 不使用
        'split_root':      './data/loop_eval',
        'derive_root':     '',   # 不使用默认 derive
        'stages':          ['export', 'split'],
        'auto_skip':       ['downsample', 'derive'],
    },
    'generic': {
        'export_root':     './data/corridor_all',
        'downsample_root': './data/corridor_balanced',
        'split_root':      './data/corridor',
        'derive_root':     './data/stage1',
        'stages':          ['export', 'downsample', 'split', 'derive'],
        'auto_skip':       [],
    },
}

DEFAULT_BAG_DIR = './data/bags'

# 通用模式到阶段的映射 (不含特殊阶段)
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
    """阶段: 从 rosbag 导出原始帧数据"""
    _print_stage_header('EXPORT', 'rosbag -> 原始帧')

    if not args.bag_dir or not os.path.isdir(args.bag_dir):
        print(f'  ✗ bag_dir 不存在或未指定: {args.bag_dir}')
        print(f'  请用 --bag_dir 指定 rosbag 文件目录')
        return False

    try:
        sys.path.insert(0, os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        from scripts.batch_export import main as batch_export_main
    except ImportError:
        try:
            from batch_export import main as batch_export_main
        except ImportError:
            print('  ✗ 无法导入 batch_export.py')
            return False

    export_argv = [
        '--bag_dir', args.bag_dir,
        '--output_dir', args.export_root,
    ]
    if getattr(args, 'img_h', None):
        export_argv.extend(['--img_h', str(args.img_h)])
    if getattr(args, 'img_w', None):
        export_argv.extend(['--img_w', str(args.img_w)])
    if getattr(args, 'odom_topic', None):
        export_argv.extend(['--odom_topic', args.odom_topic])
    if args.force:
        export_argv.append('--force')

    print(f'  命令等效: python scripts/batch_export.py '
          f'{" ".join(export_argv)}')

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
    """阶段: 智能降采样 (仅 junction / generic 适用)"""
    _print_stage_header('DOWNSAMPLE', '智能降采样')

    try:
        from scripts.downsample_corridor import run_downsample
    except ImportError:
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
        print(f'\n  ✓ Downsample 完成 -> '
              f'{os.path.abspath(args.downsample_root)}')
        return True
    except Exception as e:
        print(f'  ✗ Downsample 失败: {e}')
        return False


def stage_split(args):
    """阶段: train/val/test 划分"""
    _print_stage_header('SPLIT', 'train/val/test 划分')

    try:
        from scripts.split_corridor_runs import run_split
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from split_corridor_runs import run_split

    # split 输入取决于是否有 downsample
    src = args.downsample_root \
        if 'downsample' in args._executed_stages else args.export_root

    # straight_keep 默认使用不同的 group_by
    group_by = args.group_by
    if args.task_type == 'straight_keep' \
            and group_by == 'junction_id,turn_dir':
        group_by = 'segment_id,direction'
        print(f'  ⓘ straight_keep 自动调整 group_by -> {group_by}')

    try:
        run_split(
            src_root=src,
            dst_root=args.split_root,
            split_mode=args.split_mode,
            group_by=group_by,
            train_per_group=args.train_per_group,
            val_per_group=args.val_per_group,
            test_per_group=args.test_per_group,
            min_frames=args.min_frames,
            exclude=args.exclude,
            manifest_path=args.manifest_path,
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
    """阶段: 派生任务数据集 (junction / generic: derive_stage1)"""
    _print_stage_header('DERIVE', '派生任务数据集 (stage1)')

    try:
        from scripts.derive_stage1_datasets import main as derive_main
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from derive_stage1_datasets import main as derive_main

    derive_argv = [
        '--src_root', args.split_root,
        '--dst_root', args.derive_root,
        '--task', args.derive_task,
        '--seed', str(args.seed),
    ]
    if args.force:
        derive_argv.append('--force')
    if getattr(args, 'pre_turn_ms', None) is not None:
        derive_argv.extend(['--pre_turn_ms', str(args.pre_turn_ms)])
    if getattr(args, 'recover_ms', None) is not None:
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


def stage_derive_straight_keep(args):
    """阶段: 派生直行纠偏回归数据集"""
    _print_stage_header('DERIVE_STRAIGHT_KEEP', '直行纠偏回归数据集派生')

    try:
        from scripts.derive_straight_keep_dataset import \
            run_derive_straight_keep
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from derive_straight_keep_dataset import run_derive_straight_keep

    try:
        run_derive_straight_keep(
            src_root=args.split_root,
            dst_root=args.derive_root,
            valid_only=True,
            trim_start_ms=getattr(args, 'trim_start_ms', 500),
            trim_end_ms=getattr(args, 'trim_end_ms', 500),
            settle_window_ms=getattr(args, 'settle_window_ms', 1500),
            max_settled_frames=getattr(args, 'max_settled_frames', 20),
            copy_mode=args.copy_mode,
            seed=args.seed,
            force=args.force,
        )
        print(f'\n  ✓ Derive (straight_keep) 完成 -> '
              f'{os.path.abspath(args.derive_root)}')
        return True
    except Exception as e:
        print(f'  ✗ Derive (straight_keep) 失败: {e}')
        return False


def stage_extract_loop_windows(args):
    """阶段: 提取 loop 稀疏事件窗口"""
    _print_stage_header('EXTRACT_LOOP_WINDOWS', 'Loop 稀疏事件窗口提取')

    try:
        from scripts.extract_loop_windows import run_extract_loop_windows
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from extract_loop_windows import run_extract_loop_windows

    loop_dst = getattr(args, 'loop_sparse_root',
                       './data/loop_sparse')

    try:
        run_extract_loop_windows(
            src_root=args.split_root,
            dst_root=loop_dst,
            mode=getattr(args, 'loop_extract_mode', 'all'),
            valid_only=True,
            turn_k_consecutive=getattr(args, 'turn_k_consecutive', 3),
            turn_w_threshold=getattr(args, 'turn_w_threshold', 0.3),
            pre_turn_ms=getattr(args, 'pre_turn_ms', 2000) or 2000,
            post_turn_ms=getattr(args, 'post_turn_ms', 1200) or 1200,
            recover_ms=getattr(args, 'recover_ms', 1800) or 1800,
            stable_follow_clip_ms=getattr(
                args, 'stable_follow_clip_ms', 2000),
            stable_follow_per_segment=getattr(
                args, 'stable_follow_per_segment', 2),
            copy_mode=args.copy_mode,
            seed=args.seed,
            force=args.force,
        )
        print(f'\n  ✓ Extract loop windows 完成 -> '
              f'{os.path.abspath(loop_dst)}')
        return True
    except Exception as e:
        print(f'  ✗ Extract loop windows 失败: {e}')
        return False


def _print_stage_header(name, desc):
    print(f'\n{"=" * 72}')
    print(f'  {name}: {desc}')
    print(f'{"=" * 72}')


# ============================================================================
# 阶段函数映射
# ============================================================================

STAGE_FUNCS = {
    'export':                stage_export,
    'downsample':            stage_downsample,
    'split':                 stage_split,
    'derive':                stage_derive,
    'derive_straight_keep':  stage_derive_straight_keep,
    'extract_loop_windows':  stage_extract_loop_windows,
}


# ============================================================================
# 根据 task_type 确定实际执行阶段
# ============================================================================

def resolve_stages(args):
    """
    根据 task_type + mode + skip 确定最终执行阶段列表。

    Returns:
        list of str: 阶段名列表
    """
    tt = args.task_type
    tt_config = TASK_TYPE_DEFAULTS[tt]

    # 基础阶段来自 task_type 配置
    if args.mode == 'all':
        stages = list(tt_config['stages'])
    elif args.mode == 'export_only':
        stages = ['export']
    elif args.mode == 'prepare':
        base = list(tt_config['stages'])
        # prepare = 除了 derive 类阶段
        stages = [s for s in base
                  if not s.startswith('derive')
                  and s != 'extract_loop_windows']
    elif args.mode == 'derive_only':
        # 只执行 derive 类阶段
        base = list(tt_config['stages'])
        stages = [s for s in base if s.startswith('derive')]
        if not stages:
            stages = ['derive']
    else:
        stages = list(tt_config['stages'])

    # loop + extract_windows
    if tt == 'loop' and args.loop_extract_windows:
        if 'extract_loop_windows' not in stages:
            stages.append('extract_loop_windows')

    # 应用 auto_skip
    auto_skip = set(tt_config.get('auto_skip', []))

    # 应用用户 skip
    skip_map = {
        'export': args.skip_export,
        'downsample': args.skip_downsample,
        'split': args.skip_split,
        'derive': args.skip_derive,
        'derive_straight_keep': args.skip_derive,
        'extract_loop_windows': False,
    }

    final = []
    for s in stages:
        if s in auto_skip:
            continue
        if skip_map.get(s, False):
            continue
        final.append(s)

    return final


def apply_task_type_defaults(args):
    """
    若用户未显式指定路径，则按 task_type 设置默认路径。
    通过检查是否使用了 argparse 默认值来判断。
    """
    tt = args.task_type
    tt_config = TASK_TYPE_DEFAULTS[tt]

    # 只在用户没有显式指定时才覆盖
    # 我们通过比较当前值与 generic 默认值来判断
    generic = TASK_TYPE_DEFAULTS['generic']

    if args.export_root == generic['export_root'] and tt != 'generic':
        args.export_root = tt_config['export_root']
    if args.downsample_root == generic['downsample_root'] and tt != 'generic':
        args.downsample_root = tt_config['downsample_root'] or \
            args.downsample_root
    if args.split_root == generic['split_root'] and tt != 'generic':
        args.split_root = tt_config['split_root']
    if args.derive_root == generic['derive_root'] and tt != 'generic':
        if tt_config['derive_root']:
            args.derive_root = tt_config['derive_root']


# ============================================================================
# 主入口
# ============================================================================

def main():
    generic_defaults = TASK_TYPE_DEFAULTS['generic']

    parser = argparse.ArgumentParser(
        description='走廊导航数据集统一处理管线 '
                    '(支持 junction / straight_keep / loop / generic)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- 任务类型 ----
    parser.add_argument('--task_type', type=str, default='generic',
                        choices=['junction', 'straight_keep', 'loop',
                                 'generic'],
                        help='数据类型，决定默认处理流程和路径')

    # ---- 模式 ----
    parser.add_argument('--mode', type=str, default='all',
                        choices=['export_only', 'prepare', 'derive_only',
                                 'all'],
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
    parser.add_argument('--export_root', type=str,
                        default=generic_defaults['export_root'],
                        help='export 输出')
    parser.add_argument('--downsample_root', type=str,
                        default=generic_defaults['downsample_root'],
                        help='downsample 输出 (junction/generic 适用)')
    parser.add_argument('--split_root', type=str,
                        default=generic_defaults['split_root'],
                        help='split 输出 / derive 输入')
    parser.add_argument('--derive_root', type=str,
                        default=generic_defaults['derive_root'],
                        help='derive 输出')

    # ---- Export 参数 ----
    parser.add_argument('--img_h', type=int, default=None,
                        help='导出图片高度 (None=原始)')
    parser.add_argument('--img_w', type=int, default=None,
                        help='导出图片宽度 (None=原始)')
    parser.add_argument('--odom_topic', type=str, default=None,
                        help='里程计话题名称 (默认 /odom_raw)')

    # ---- Downsample 参数 (junction/generic) ----
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
                        help='分组键 (straight_keep 自动调整为 segment_id,direction)')
    parser.add_argument('--train_per_group', type=int, default=5)
    parser.add_argument('--val_per_group', type=int, default=1)
    parser.add_argument('--test_per_group', type=int, default=1)
    parser.add_argument('--manifest_path', type=str, default=None,
                        help='run manifest CSV 路径 (含 group 分组信息)')
    parser.add_argument('--min_frames', type=int, default=10,
                        help='最少帧数，低于该值的 run 将被跳过')

    # ---- Derive 参数 (junction/generic) ----
    parser.add_argument('--derive_task', type=str, default='all',
                        choices=['action3_balanced', 'junction_lr',
                                 'stage4', 'all'],
                        help='派生任务 (junction/generic)')
    parser.add_argument('--pre_turn_ms', type=float, default=None,
                        help='turn_on 前窗口 (ms)')
    parser.add_argument('--recover_ms', type=float, default=None,
                        help='turn_off 后 recover 窗口 (ms)')

    # ---- Derive 参数 (straight_keep) ----
    parser.add_argument('--trim_start_ms', type=float, default=500,
                        help='straight_keep: 去掉开头缓冲 (ms)')
    parser.add_argument('--trim_end_ms', type=float, default=500,
                        help='straight_keep: 去掉结尾缓冲 (ms)')
    parser.add_argument('--settle_window_ms', type=float, default=1500,
                        help='straight_keep: Settled 最小窗口 (ms)')
    parser.add_argument('--max_settled_frames', type=int, default=20,
                        help='straight_keep: Settled 最大帧数')

    # ---- Loop 参数 ----
    parser.add_argument('--loop_extract_windows', action='store_true',
                        help='loop: split 后额外提取稀疏事件窗口')
    parser.add_argument('--loop_sparse_root', type=str,
                        default='./data/loop_sparse',
                        help='loop: 窗口提取输出目录')
    parser.add_argument('--loop_extract_mode', type=str, default='all',
                        choices=['junction_windows', 'stage_windows',
                                 'sparse_follow', 'all'],
                        help='loop: 窗口提取模式')
    parser.add_argument('--turn_k_consecutive', type=int, default=3,
                        help='loop: turn 检测连续帧数')
    parser.add_argument('--turn_w_threshold', type=float, default=0.3,
                        help='loop: turn 检测 angular_z 阈值')
    parser.add_argument('--post_turn_ms', type=float, default=1200,
                        help='loop: turn_off 后窗口 (ms)')
    parser.add_argument('--stable_follow_clip_ms', type=float, default=2000,
                        help='loop: follow 子片段长度 (ms)')
    parser.add_argument('--stable_follow_per_segment', type=int, default=2,
                        help='loop: 每段 follow 子片段数')

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

    # ======================== 按 task_type 设置默认路径 ========================
    apply_task_type_defaults(args)

    # ======================== 确定执行阶段 ========================
    args._executed_stages = []   # 用于 split 判断输入源
    stages = resolve_stages(args)

    if not stages:
        print('没有需要执行的阶段 (全部被跳过)')
        return

    # ======================== Banner ========================
    tt = args.task_type
    tt_label = {
        'junction': '路口转弯 (junction)',
        'straight_keep': '长直行纠偏 (straight_keep)',
        'loop': '闭环 loop',
        'generic': '通用流程 (generic)',
    }[tt]

    print('=' * 72)
    print('  走廊导航数据集统一处理管线')
    print('=' * 72)
    print(f'  任务类型:  {tt_label}')
    print(f'  管线模式:  {args.mode}')
    print(f'  执行阶段:  {" -> ".join(stages)}')
    print(f'  路径:')
    if 'export' in stages:
        print(f'    bags      : {os.path.abspath(args.bag_dir)}')
        print(f'    export    : {os.path.abspath(args.export_root)}')
    if 'downsample' in stages:
        print(f'    downsample: {os.path.abspath(args.downsample_root)}')
    if 'split' in stages:
        print(f'    split     : {os.path.abspath(args.split_root)}')
    for s in stages:
        if s.startswith('derive'):
            print(f'    derive    : {os.path.abspath(args.derive_root)}')
            break
    if 'extract_loop_windows' in stages:
        print(f'    loop_sparse: '
              f'{os.path.abspath(args.loop_sparse_root)}')
    print(f'  Force:     {args.force}')
    print(f'  Seed:      {args.seed}')

    # 特殊提示
    if tt == 'straight_keep':
        print()
        print('  ⓘ straight_keep 数据默认不使用 downsample_corridor.py')
        print('    (turn-context 降采样规则不适合直行纠偏数据)')
        print('  ⓘ 推荐使用 manifest 按 segment_id,direction,station_id,condition 分组划分')
        if hasattr(args, 'manifest_path') and args.manifest_path:
            print(f'    → 已指定 manifest: {args.manifest_path}')
        else:
            print('    → 未指定 manifest，将按 run 目录名自动推断分组')
        if hasattr(args, 'odom_topic') and args.odom_topic:
            print(f'  ⓘ odom 话题: {args.odom_topic}')
        else:
            print('  ⓘ odom 话题: /odom_raw (默认)')
    if tt == 'loop' and not args.loop_extract_windows:
        print()
        print('  ⓘ 若需提取稀疏事件窗口，请加 --loop_extract_windows')

    print('=' * 72)

    if args.dry_run:
        print('\n  [dry_run] 仅预览，不执行')
        return

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
        args._executed_stages.append(stage)

        if not ok:
            print(f'\n  ✗ 阶段 {stage} 失败 (耗时 {elapsed:.1f}s)')
            print(f'  后续阶段已跳过')
            break
        else:
            print(f'\n  ✓ 阶段 {stage} 完成 (耗时 {elapsed:.1f}s)')

    total_elapsed = time.time() - t0

    # ======================== 总结 ========================
    print(f'\n{"=" * 72}')
    print(f'  管线执行总结  [{tt_label}]')
    print(f'{"=" * 72}')
    for stage, info in results.items():
        status = '✓' if info['success'] else '✗'
        print(f'  [{status}] {stage:25s}  {info["elapsed_s"]:6.1f}s')
    print(f'  {"─" * 40}')
    print(f'  总耗时: {total_elapsed:.1f}s')

    # 保存管线日志
    pipeline_log = {
        'timestamp': datetime.now().isoformat(),
        'task_type': tt,
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

    # 确定日志保存目录
    log_dir = args.split_root
    for s in ['derive', 'derive_straight_keep']:
        if s in results:
            log_dir = args.derive_root
            break
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
