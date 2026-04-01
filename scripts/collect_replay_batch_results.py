"""
批量 Replay 结果汇总脚本
========================

功能：
  遍历指定 replay 根目录下所有子目录，读取每个子目录中的 replay_summary.json，
  汇总为一张系统级总表（CSV + Markdown），并在终端打印关键统计摘要。

输入：
  --exp_root   replay 根目录，例如 ./results/replay_batch_valid
               该目录下每个子目录应包含 replay_summary.json

输出：
  --out_csv    汇总 CSV 表格路径（默认 <exp_root>/replay_batch_summary.csv）
  --out_md     汇总 Markdown 简报路径（默认 <exp_root>/replay_batch_summary.md）

示例命令：
  python scripts/collect_replay_batch_results.py ^
      --exp_root results/replay_batch_valid

  python scripts/collect_replay_batch_results.py ^
      --exp_root results/replay_batch_valid ^
      --out_csv  results/my_summary.csv ^
      --out_md   results/my_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 常量：汇总表需要收集的字段列表
# ---------------------------------------------------------------------------
COLLECT_FIELDS = [
    'run_name',
    'gt_turn_dir',
    'first_locked_turn_dir',
    'most_frequent_locked_turn_dir',
    'corrected_eval_dir',
    'exit_mode',
    'turn_dir_match',
    'corrected_turn_dir_match',
    'has_late_correction',
    'num_turn_entries',
    'num_recover_entries',
    'first_turn_step',
    'first_recover_step',
    'turn_duration_steps',
    'recover_duration_steps',
    'num_clip_applied',
    'num_turn_timeout_exits',
    'num_recover_signal_exits',
    'num_low_turn_low_omega_exits',
    'num_soft_exit_exits',
    'unique_state_sequence',
    'used_valid_only',
    'original_total_steps',
    'used_total_steps',
    'skipped_invalid_steps',
]


def _safe_get(d: Dict[str, Any], key: str, default: Any = '') -> Any:
    """安全地从字典中取值；若 key 不存在则返回 default。"""
    val = d.get(key, default)
    if val is None:
        return default
    return val


def _extract_run_name(summary: Dict[str, Any], subdir_name: str) -> str:
    """
    提取 run_name：
      1) 优先使用 summary 中的 run_dir 的 basename
      2) 否则使用子目录名
    """
    run_dir = summary.get('run_dir', '')
    if run_dir:
        return os.path.basename(run_dir)
    return subdir_name


def _format_sequence(seq: Any) -> str:
    """将 unique_state_sequence 列表格式化为可读字符串。"""
    if isinstance(seq, list):
        return ' -> '.join(str(s) for s in seq)
    return str(seq) if seq else ''


def _normalize_turn_dir(val: Any) -> str:
    """将方向字段规范化为 Left/Right/''。"""
    s = str(val).strip()
    if s in ('Left', 'Right'):
        return s
    return ''


def _load_summary(json_path: str) -> Optional[Dict[str, Any]]:
    """加载单个 replay_summary.json，失败返回 None。"""
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        print(f'  [警告] 无法解析 {json_path}: {e}')
        return None


def _collect_all_runs(exp_root: str) -> List[Dict[str, Any]]:
    """
    遍历 exp_root 下所有子目录，收集 replay_summary.json。
    返回行记录列表，每行对应一个 run 的汇总字段。
    """
    if not os.path.isdir(exp_root):
        print(f'[错误] 目录不存在: {exp_root}')
        sys.exit(1)

    # 列出所有子目录（按字典序排序）
    subdirs = sorted([
        d for d in os.listdir(exp_root)
        if os.path.isdir(os.path.join(exp_root, d))
    ])

    rows: List[Dict[str, Any]] = []
    skipped = 0

    for sd in subdirs:
        json_path = os.path.join(exp_root, sd, 'replay_summary.json')
        summary = _load_summary(json_path)
        if summary is None:
            skipped += 1
            continue

        # 提取每一个字段，缺失则留空
        run_name = _extract_run_name(summary, sd)
        unique_seq = _safe_get(summary, 'unique_state_sequence', [])
        gt_turn_dir = _normalize_turn_dir(_safe_get(summary, 'gt_turn_dir', ''))
        first_locked_turn_dir = _normalize_turn_dir(
            _safe_get(summary, 'first_locked_turn_dir', '')
        )
        most_frequent_locked_turn_dir = _normalize_turn_dir(
            _safe_get(summary, 'most_frequent_locked_turn_dir', '')
        )

        # 辅助分析指标：优先使用 most_frequent_locked_turn_dir 评估后期纠正是否成功
        corrected_eval_dir = (
            most_frequent_locked_turn_dir
            if most_frequent_locked_turn_dir
            else first_locked_turn_dir
        )
        if gt_turn_dir and corrected_eval_dir:
            corrected_turn_dir_match: Any = (corrected_eval_dir == gt_turn_dir)
        else:
            corrected_turn_dir_match = ''
        has_late_correction = bool(
            most_frequent_locked_turn_dir
            and first_locked_turn_dir != most_frequent_locked_turn_dir
        )

        # 退出模式相关计数（缺失时安全兜底为 0）
        num_turn_timeout_exits = _to_int(_safe_get(summary, 'num_turn_timeout_exits', 0)) or 0
        num_recover_signal_exits = _to_int(_safe_get(summary, 'num_recover_signal_exits', 0)) or 0
        num_low_turn_low_omega_exits = _to_int(_safe_get(summary, 'num_low_turn_low_omega_exits', 0)) or 0
        num_soft_exit_exits = _to_int(_safe_get(summary, 'num_soft_exit_exits', 0)) or 0

        # 兼容多个计数同时 > 0，按优先级确定主退出模式
        if num_soft_exit_exits > 0:
            exit_mode = 'soft_exit'
        elif num_recover_signal_exits > 0:
            exit_mode = 'recover_signal'
        elif num_low_turn_low_omega_exits > 0:
            exit_mode = 'low_turn_low_omega'
        elif num_turn_timeout_exits > 0:
            exit_mode = 'timeout'
        else:
            exit_mode = 'unknown'

        row: Dict[str, Any] = {
            'run_name':                      run_name,
            'gt_turn_dir':                   gt_turn_dir,
            'first_locked_turn_dir':         first_locked_turn_dir,
            'most_frequent_locked_turn_dir': most_frequent_locked_turn_dir,
            'corrected_eval_dir':            corrected_eval_dir,
            'exit_mode':                     exit_mode,
            'turn_dir_match':                _safe_get(summary, 'turn_dir_match', ''),
            'corrected_turn_dir_match':      corrected_turn_dir_match,
            'has_late_correction':           has_late_correction,
            'num_turn_entries':              _safe_get(summary, 'num_turn_entries', ''),
            'num_recover_entries':           _safe_get(summary, 'num_recover_entries', ''),
            'first_turn_step':               _safe_get(summary, 'first_turn_step', ''),
            'first_recover_step':            _safe_get(summary, 'first_recover_step', ''),
            'turn_duration_steps':           _safe_get(summary, 'turn_duration_steps', ''),
            'recover_duration_steps':        _safe_get(summary, 'recover_duration_steps', ''),
            'num_clip_applied':              _safe_get(summary, 'num_clip_applied', ''),
            'num_turn_timeout_exits':        num_turn_timeout_exits,
            'num_recover_signal_exits':      num_recover_signal_exits,
            'num_low_turn_low_omega_exits':  num_low_turn_low_omega_exits,
            'num_soft_exit_exits':           num_soft_exit_exits,
            'unique_state_sequence':         _format_sequence(unique_seq),
            'used_valid_only':               _safe_get(summary, 'used_valid_only', ''),
            'original_total_steps':          _safe_get(summary, 'original_total_steps', ''),
            'used_total_steps':              _safe_get(summary, 'used_total_steps', ''),
            'skipped_invalid_steps':         _safe_get(summary, 'skipped_invalid_steps', ''),
        }

        # 额外保留原始 summary 引用，用于聚合统计
        row['_raw'] = summary
        rows.append(row)

    if skipped > 0:
        print(f'  [信息] 跳过了 {skipped} 个子目录（未找到 replay_summary.json）')

    return rows


# ---------------------------------------------------------------------------
# 聚合统计
# ---------------------------------------------------------------------------
def _compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """根据汇总行列表计算系统级聚合统计。"""
    total_runs = len(rows)
    if total_runs == 0:
        return {'total_runs': 0}

    # turn_dir_match_rate：turn_dir_match == True 的比例
    match_count = sum(1 for r in rows if r.get('turn_dir_match') is True)
    valid_match_count = sum(1 for r in rows if r.get('turn_dir_match') in (True, False))
    turn_dir_match_rate = match_count / valid_match_count if valid_match_count > 0 else None

    # corrected_turn_dir_match_rate（辅助分析指标）：
    # 优先按 corrected_eval_dir（most_frequent 优先）评估与 gt 的匹配比例
    corrected_match_count = sum(1 for r in rows if r.get('corrected_turn_dir_match') is True)
    valid_corrected_match_count = sum(
        1 for r in rows if r.get('corrected_turn_dir_match') in (True, False)
    )
    corrected_turn_dir_match_rate = (
        corrected_match_count / valid_corrected_match_count
        if valid_corrected_match_count > 0 else None
    )

    # late_correction_rate：发生“首锁与后期主导方向不同”的比例（辅助分析）
    late_correction_count = sum(1 for r in rows if r.get('has_late_correction') is True)
    valid_late_correction_count = sum(1 for r in rows if r.get('has_late_correction') in (True, False))
    late_correction_rate = (
        late_correction_count / valid_late_correction_count
        if valid_late_correction_count > 0 else None
    )

    # single_turn_success_rate：num_turn_entries == 1
    single_turn_count = sum(1 for r in rows if _to_int(r.get('num_turn_entries')) == 1)
    single_turn_success_rate = single_turn_count / total_runs

    # single_recover_success_rate：num_recover_entries == 1
    single_recover_count = sum(1 for r in rows if _to_int(r.get('num_recover_entries')) == 1)
    single_recover_success_rate = single_recover_count / total_runs

    # full_sequence_success_rate：unique_state_sequence 同时包含 APPROACH, TURN, RECOVER
    full_seq_count = 0
    for r in rows:
        seq_str = str(r.get('unique_state_sequence', ''))
        if all(kw in seq_str for kw in ('APPROACH', 'TURN', 'RECOVER')):
            full_seq_count += 1
    full_sequence_success_rate = full_seq_count / total_runs

    # 退出模式触发率：直接使用 replay_summary 中的退出计数字段
    timeout_runs: List[str] = []
    recover_signal_runs: List[str] = []
    low_turn_low_omega_runs: List[str] = []
    soft_exit_runs: List[str] = []

    for r in rows:
        raw = r.get('_raw', {})
        if not isinstance(raw, dict):
            raw = {}
        run_name = str(r.get('run_name', ''))

        num_turn_timeout_exits = _to_int(raw.get('num_turn_timeout_exits', 0)) or 0
        num_recover_signal_exits = _to_int(raw.get('num_recover_signal_exits', 0)) or 0
        num_low_turn_low_omega_exits = _to_int(raw.get('num_low_turn_low_omega_exits', 0)) or 0
        num_soft_exit_exits = _to_int(raw.get('num_soft_exit_exits', 0)) or 0

        timeout_run = (num_turn_timeout_exits > 0)
        recover_signal_run = (num_recover_signal_exits > 0)
        low_turn_low_omega_run = (num_low_turn_low_omega_exits > 0)
        soft_exit_run = (num_soft_exit_exits > 0)

        if timeout_run:
            timeout_runs.append(run_name)
        if recover_signal_run:
            recover_signal_runs.append(run_name)
        if low_turn_low_omega_run:
            low_turn_low_omega_runs.append(run_name)
        if soft_exit_run:
            soft_exit_runs.append(run_name)

    timeout_trigger_rate = len(timeout_runs) / total_runs
    recover_signal_rate = len(recover_signal_runs) / total_runs
    low_turn_low_omega_rate = len(low_turn_low_omega_runs) / total_runs
    soft_exit_rate = len(soft_exit_runs) / total_runs

    # mean_turn_duration_steps
    turn_durations = [_to_float(r.get('turn_duration_steps'))
                      for r in rows if _to_float(r.get('turn_duration_steps')) is not None]
    mean_turn_duration = sum(turn_durations) / len(turn_durations) if turn_durations else None

    # mean_recover_duration_steps
    recover_durations = [_to_float(r.get('recover_duration_steps'))
                         for r in rows if _to_float(r.get('recover_duration_steps')) is not None]
    mean_recover_duration = sum(recover_durations) / len(recover_durations) if recover_durations else None

    return {
        'total_runs':                  total_runs,
        'turn_dir_match_rate':         turn_dir_match_rate,
        'corrected_turn_dir_match_rate': corrected_turn_dir_match_rate,
        'late_correction_rate':        late_correction_rate,
        'single_turn_success_rate':    single_turn_success_rate,
        'single_recover_success_rate': single_recover_success_rate,
        'full_sequence_success_rate':  full_sequence_success_rate,
        'timeout_trigger_rate':        timeout_trigger_rate,
        'recover_signal_rate':         recover_signal_rate,
        'low_turn_low_omega_rate':     low_turn_low_omega_rate,
        'soft_exit_rate':              soft_exit_rate,
        'timeout_runs':                timeout_runs,
        'recover_signal_runs':         recover_signal_runs,
        'low_turn_low_omega_runs':     low_turn_low_omega_runs,
        'soft_exit_runs':              soft_exit_runs,
        'mean_turn_duration_steps':    mean_turn_duration,
        'mean_recover_duration_steps': mean_recover_duration,
    }


def _to_int(val: Any) -> Optional[int]:
    """安全转 int。"""
    if val is None or val == '':
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _to_float(val: Any) -> Optional[float]:
    """安全转 float。"""
    if val is None or val == '':
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fmt_pct(val: Optional[float]) -> str:
    """格式化百分比。"""
    if val is None:
        return 'N/A'
    return f'{val * 100:.1f}%'


def _fmt_float(val: Optional[float], decimals: int = 2) -> str:
    """格式化浮点数。"""
    if val is None:
        return 'N/A'
    return f'{val:.{decimals}f}'


def _fmt_run_list(val: Any) -> str:
    """格式化 run 列表。"""
    if isinstance(val, list) and len(val) > 0:
        return ', '.join(str(x) for x in val)
    return 'None'


# ---------------------------------------------------------------------------
# 输出：CSV
# ---------------------------------------------------------------------------
def _write_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    """写入 CSV 汇总表。"""
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLLECT_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'  [✓] CSV 已保存: {out_csv}')


# ---------------------------------------------------------------------------
# 输出：Markdown
# ---------------------------------------------------------------------------
def _write_md(rows: List[Dict[str, Any]], stats: Dict[str, Any], out_md: str) -> None:
    """写入 Markdown 简报。"""
    os.makedirs(os.path.dirname(out_md) or '.', exist_ok=True)

    lines: List[str] = []
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines.append(f'# Replay Batch Summary')
    lines.append('')
    lines.append(f'> 生成时间: {now_str}')
    lines.append(f'> 总 runs 数: {stats.get("total_runs", 0)}')
    lines.append('')

    # ---- 聚合统计表 ----
    lines.append('## 聚合统计')
    lines.append('')
    lines.append('| 指标 | 值 |')
    lines.append('|------|-----|')
    lines.append(f'| total_runs | {stats.get("total_runs", 0)} |')
    lines.append(f'| turn_dir_match_rate (首锁正确率, 主指标) | {_fmt_pct(stats.get("turn_dir_match_rate"))} |')
    lines.append(f'| corrected_turn_dir_match_rate (后期纠正后匹配率, 辅助指标) | {_fmt_pct(stats.get("corrected_turn_dir_match_rate"))} |')
    lines.append(f'| late_correction_rate (发生后期纠正比例, 辅助指标) | {_fmt_pct(stats.get("late_correction_rate"))} |')
    lines.append(f'| single_turn_success_rate | {_fmt_pct(stats.get("single_turn_success_rate"))} |')
    lines.append(f'| single_recover_success_rate | {_fmt_pct(stats.get("single_recover_success_rate"))} |')
    lines.append(f'| full_sequence_success_rate | {_fmt_pct(stats.get("full_sequence_success_rate"))} |')
    lines.append(f'| timeout_trigger_rate | {_fmt_pct(stats.get("timeout_trigger_rate"))} |')
    lines.append(f'| recover_signal_rate | {_fmt_pct(stats.get("recover_signal_rate"))} |')
    lines.append(f'| low_turn_low_omega_rate | {_fmt_pct(stats.get("low_turn_low_omega_rate"))} |')
    lines.append(f'| soft_exit_rate | {_fmt_pct(stats.get("soft_exit_rate"))} |')
    lines.append(f'| mean_turn_duration_steps | {_fmt_float(stats.get("mean_turn_duration_steps"))} |')
    lines.append(f'| mean_recover_duration_steps | {_fmt_float(stats.get("mean_recover_duration_steps"))} |')
    lines.append('')

    # ---- 逐 run 明细表 ----
    lines.append('## 逐 Run 明细')
    lines.append('')

    # 选择展示在 Markdown 表中的关键列（过多列会导致表格不可读）
    md_cols = [
        ('run_name',                      'Run'),
        ('gt_turn_dir',                   'GT Dir'),
        ('first_locked_turn_dir',         '1st Lock'),
        ('most_frequent_locked_turn_dir', 'Most Freq Lock'),
        ('corrected_eval_dir',            'Corr Eval Dir'),
        ('turn_dir_match',                '1st Match'),
        ('corrected_turn_dir_match',      'Corr Match'),
        ('has_late_correction',           'Late Corr'),
        ('exit_mode',                     'Exit Mode'),
        ('num_turn_timeout_exits',        'Timeout Cnt'),
        ('num_recover_signal_exits',      'Recover Sig Cnt'),
        ('num_low_turn_low_omega_exits',  'LowTurnLowOmega Cnt'),
        ('num_soft_exit_exits',           'SoftExit Cnt'),
        ('num_turn_entries',              '#Turn'),
        ('num_recover_entries',           '#Recov'),
        ('first_turn_step',              '1st Turn'),
        ('first_recover_step',           '1st Recov'),
        ('turn_duration_steps',          'T Dur'),
        ('recover_duration_steps',       'R Dur'),
        ('num_clip_applied',             '#Clip'),
        ('used_total_steps',             'Steps'),
        ('unique_state_sequence',        'State Seq'),
    ]

    # 表头
    header = '| ' + ' | '.join(h for _, h in md_cols) + ' |'
    sep = '| ' + ' | '.join('---' for _ in md_cols) + ' |'
    lines.append(header)
    lines.append(sep)

    for r in rows:
        cells = []
        for field, _ in md_cols:
            val = r.get(field, '')
            # 布尔值转显示
            if isinstance(val, bool):
                val = '✓' if val else '✗'
            cells.append(str(val))
        lines.append('| ' + ' | '.join(cells) + ' |')

    lines.append('')
    lines.append('### Exit Mode Breakdown')
    lines.append(f'- timeout: {_fmt_run_list(stats.get("timeout_runs"))}')
    lines.append(f'- recover_signal: {_fmt_run_list(stats.get("recover_signal_runs"))}')
    lines.append(f'- low_turn_low_omega: {_fmt_run_list(stats.get("low_turn_low_omega_runs"))}')
    lines.append(f'- soft_exit: {_fmt_run_list(stats.get("soft_exit_runs"))}')
    lines.append('')
    lines.append('---')
    lines.append(f'*由 `collect_replay_batch_results.py` 自动生成*')
    lines.append('')

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  [✓] Markdown 已保存: {out_md}')


# ---------------------------------------------------------------------------
# 终端打印
# ---------------------------------------------------------------------------
def _print_stats(stats: Dict[str, Any]) -> None:
    """在终端打印关键统计。"""
    print()
    print('=' * 64)
    print('  Replay Batch 聚合统计')
    print('=' * 64)
    print(f'  total_runs                  : {stats.get("total_runs", 0)}')
    print(f'  turn_dir_match_rate         : {_fmt_pct(stats.get("turn_dir_match_rate"))}  (主指标: 首锁正确率)')
    print(f'  corrected_turn_dir_match_rate: {_fmt_pct(stats.get("corrected_turn_dir_match_rate"))}  (辅助: 后期纠正后匹配率)')
    print(f'  late_correction_rate        : {_fmt_pct(stats.get("late_correction_rate"))}  (辅助: 发生后期纠正比例)')
    print(f'  single_turn_success_rate    : {_fmt_pct(stats.get("single_turn_success_rate"))}')
    print(f'  single_recover_success_rate : {_fmt_pct(stats.get("single_recover_success_rate"))}')
    print(f'  full_sequence_success_rate  : {_fmt_pct(stats.get("full_sequence_success_rate"))}')
    print(f'  timeout_trigger_rate        : {_fmt_pct(stats.get("timeout_trigger_rate"))}')
    print(f'  recover_signal_rate         : {_fmt_pct(stats.get("recover_signal_rate"))}')
    print(f'  low_turn_low_omega_rate     : {_fmt_pct(stats.get("low_turn_low_omega_rate"))}')
    print(f'  soft_exit_rate              : {_fmt_pct(stats.get("soft_exit_rate"))}')
    print(f'  timeout_runs                : {_fmt_run_list(stats.get("timeout_runs"))}')
    print(f'  recover_signal_runs         : {_fmt_run_list(stats.get("recover_signal_runs"))}')
    print(f'  low_turn_low_omega_runs     : {_fmt_run_list(stats.get("low_turn_low_omega_runs"))}')
    print(f'  soft_exit_runs              : {_fmt_run_list(stats.get("soft_exit_runs"))}')
    print(f'  mean_turn_duration_steps    : {_fmt_float(stats.get("mean_turn_duration_steps"))}')
    print(f'  mean_recover_duration_steps : {_fmt_float(stats.get("mean_recover_duration_steps"))}')
    print('=' * 64)
    print()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description='汇总批量 replay 的 replay_summary.json，生成系统级总表与简报'
    )
    parser.add_argument('--exp_root', type=str, required=True,
                        help='replay 根目录，其下每个子目录应包含 replay_summary.json')
    parser.add_argument('--out_csv', type=str, default='',
                        help='输出 CSV 路径（默认 <exp_root>/replay_batch_summary.csv）')
    parser.add_argument('--out_md', type=str, default='',
                        help='输出 Markdown 路径（默认 <exp_root>/replay_batch_summary.md）')
    args = parser.parse_args()

    exp_root = os.path.abspath(args.exp_root)
    out_csv = args.out_csv if args.out_csv else os.path.join(exp_root, 'replay_batch_summary.csv')
    out_md = args.out_md if args.out_md else os.path.join(exp_root, 'replay_batch_summary.md')

    print(f'[Collect] 开始汇总批量 replay 结果')
    print(f'  exp_root : {exp_root}')
    print(f'  out_csv  : {out_csv}')
    print(f'  out_md   : {out_md}')
    print()

    # 1) 收集所有 run 的 summary
    rows = _collect_all_runs(exp_root)
    if not rows:
        print('[错误] 未找到任何有效的 replay_summary.json，请检查 exp_root 目录结构。')
        sys.exit(1)

    print(f'  [信息] 成功收集 {len(rows)} 个 run 的汇总数据')

    # 2) 聚合统计
    stats = _compute_stats(rows)

    # 3) 输出 CSV
    _write_csv(rows, out_csv)

    # 4) 输出 Markdown
    _write_md(rows, stats, out_md)

    # 5) 终端打印
    _print_stats(stats)

    print('[Collect] 完成！')


if __name__ == '__main__':
    main()
