"""
实验结果自动汇总脚本
========================================
自动扫描多个实验目录，读取训练/测试结果，生成 CSV 和 Markdown 表格。

支持的实验类型:
  - action3_balanced (三分类)
  - junction_lr (左右判定)
  - stage3 (三阶段)
  - stage4 (四阶段)
  - straight_keep_reg (直行纠偏回归)
  - 其他自定义实验

自动读取的文件 (优先级从高到低):
  1. final_test_metrics.json — 最终测试结果 (含 config)
  2. metrics.json            — 训练过程中保存的指标
  3. best_model.ckpt         — checkpoint 中的 config / epoch 信息
  4. pipeline_log.json       — 数据管线日志

输出:
  - ablation_summary.csv   — 完整字段 CSV
  - ablation_summary.md    — Markdown 论文表格

用法:
  # 自动递归扫描
  python scripts/collect_experiment_results.py \\
      --exp_root ./results \\
      --recursive

  # 指定输出路径
  python scripts/collect_experiment_results.py \\
      --exp_root ./results \\
      --out_csv ./results/ablation_summary.csv \\
      --out_md ./results/ablation_summary.md \\
      --recursive --top_k 10
"""

import os
import sys
import csv
import json
import argparse
from collections import defaultdict


# ============================================================================
# 常量
# ============================================================================

# 汇总表中的字段顺序
SUMMARY_FIELDS = [
    'exp_name',
    'dataset',
    'task_name',
    'neuron_type',
    'residual_mode',
    'T',
    'img_h',
    'img_w',
    'seq_len',
    'stride',
    'best_epoch',
    'best_val_acc',
    'best_val_mae',
    'test_acc',
    'test_mae',
    'test_rmse',
    'zero_baseline_mae',
    'zero_baseline_rmse',
    'mae_gain_vs_zero_pct',
    'rmse_gain_vs_zero_pct',
    'correcting_mae',
    'correcting_rmse',
    'settled_mae',
    'settled_rmse',
    'test_loss',
    'test_spike_rate',
    'test_sparsity',
    'test_spikes_per_image',
    'source',
]

# Markdown 表格展示的精简列 (分类任务)
MD_FIELDS_CLS = [
    'exp_name', 'neuron_type', 'residual_mode', 'T',
    'best_val_acc', 'test_acc', 'test_loss',
    'test_spike_rate', 'test_sparsity',
]

# Markdown 表格展示的精简列 (回归任务)
MD_FIELDS_REG = [
    'exp_name', 'neuron_type', 'residual_mode', 'T',
    'best_val_mae', 'test_mae', 'test_rmse',
    'mae_gain_vs_zero_pct', 'rmse_gain_vs_zero_pct',
    'correcting_mae', 'settled_mae',
    'test_spike_rate', 'test_sparsity',
]

# 字段格式
FIELD_FMT = {
    'best_val_acc': '.4f',
    'best_val_mae': '.4f',
    'test_acc': '.4f',
    'test_mae': '.4f',
    'test_rmse': '.4f',
    'zero_baseline_mae': '.4f',
    'zero_baseline_rmse': '.4f',
    'mae_gain_vs_zero_pct': '.2f',
    'rmse_gain_vs_zero_pct': '.2f',
    'correcting_mae': '.4f',
    'correcting_rmse': '.4f',
    'settled_mae': '.4f',
    'settled_rmse': '.4f',
    'test_loss': '.4f',
    'test_spike_rate': '.4f',
    'test_sparsity': '.4f',
    'test_spikes_per_image': '.0f',
}


# ============================================================================
# 读取辅助函数
# ============================================================================

def _load_json(path):
    """安全加载 JSON 文件"""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _load_ckpt_meta(path):
    """
    从 PyTorch checkpoint 中提取元信息 (不加载权重)。

    尝试用 torch.load 的 map_location='cpu' 加载，
    若 torch 不可用则跳过。
    """
    try:
        import torch
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        meta = {}
        for key in ('epoch', 'max_test_acc', 'max_val_acc',
                     'min_val_mae', 'config', 'exp_name'):
            if key in ckpt:
                meta[key] = ckpt[key]
        return meta
    except Exception:
        return None


def _extract_config_fields(config):
    """从 config dict 中提取关键字段"""
    if not config or not isinstance(config, dict):
        return {}

    fields = {}

    # 直接字段映射
    direct = {
        'neuron_type': ['neuron', 'neuron_type'],
        'residual_mode': ['residual_mode', 'res_mode'],
        'T': ['T', 'time_steps', 'timesteps'],
        'img_h': ['img_h', 'image_h', 'height'],
        'img_w': ['img_w', 'image_w', 'width'],
        'seq_len': ['seq_len', 'sequence_length'],
        'stride': ['stride'],
        'dataset': ['dataset', 'data_name'],
        'task_name': ['task_name', 'task'],
    }

    for out_key, candidates in direct.items():
        for cand in candidates:
            if cand in config:
                fields[out_key] = config[cand]
                break

    return fields


def _is_missing(v):
    """判断字段是否缺失（None 或空字符串）。"""
    return v is None or (isinstance(v, str) and v.strip() == '')


def _infer_meta_from_name(name):
    """
    从实验名/目录名启发式推断 dataset / task_name。
    兼容以下命名模式：
      - corridor_task_stage3_*
      - corridor_task_stage4_*
      - corridor_task_junction_lr_*
      - corridor_task_action3_balanced_*
      - corridor_regression_*
    """
    s = str(name or '').lower()
    out = {}

    # 优先匹配最明确的 corridor_task 前缀模式
    if 'corridor_task_action3_balanced_' in s:
        out['dataset'] = 'corridor_task'
        out['task_name'] = 'action3_balanced'
        return out
    if 'corridor_task_junction_lr_' in s:
        out['dataset'] = 'corridor_task'
        out['task_name'] = 'junction_lr'
        return out
    if 'corridor_task_stage3_' in s:
        out['dataset'] = 'corridor_task'
        out['task_name'] = 'stage3'
        return out
    if 'corridor_task_stage4_' in s:
        out['dataset'] = 'corridor_task'
        out['task_name'] = 'stage4'
        return out

    # 回归命名模式
    if 'corridor_regression_' in s:
        out['dataset'] = 'corridor'
        out['task_name'] = 'straight_keep_reg'
        return out

    # 通用 token 兜底（用于 exp_name 中明确包含 task token 的情况）
    for tn in ('action3_balanced', 'junction_lr', 'stage3', 'stage4'):
        if tn in s:
            out['task_name'] = tn
            break

    return out


# ============================================================================
# 单实验读取
# ============================================================================

def read_experiment(exp_dir):
    """
    读取单个实验目录的结果。

    优先级:
      1. final_test_metrics.json (最终独立测试)
      2. metrics.json (训练过程中的测试)
      3. best_model.ckpt (checkpoint 元信息)

    Returns:
        dict 或 None
    """
    exp_name = os.path.basename(exp_dir)
    record = {'exp_name': exp_name, 'source': ''}

    sources = []

    # ---- 1) final_test_metrics.json ----
    ftm = _load_json(os.path.join(exp_dir, 'final_test_metrics.json'))
    if ftm:
        sources.append('final_test')
        record['exp_name'] = ftm.get('exp_name', exp_name)

        # 测试指标
        for key in ('test_acc', 'test_mae', 'test_rmse', 'test_loss',
                     'test_spike_rate', 'test_sparsity',
                     'test_spikes_per_image', 'test_samples',
                     'zero_baseline_mae', 'zero_baseline_rmse'):
            if key in ftm:
                record[key] = ftm[key]

        # best_val_xxx
        for key in ('best_val_acc', 'best_val_mae', 'best_epoch'):
            if key in ftm:
                record[key] = ftm[key]

        # phase_stats (straight_keep_reg 回归任务)
        ps = ftm.get('phase_stats')
        if ps and isinstance(ps, dict):
            for phase, prefix in (('Correcting', 'correcting'),
                                  ('Settled', 'settled')):
                pdata = ps.get(phase, {})
                if isinstance(pdata, dict):
                    for metric in ('mae', 'rmse'):
                        if metric in pdata:
                            record[f'{prefix}_{metric}'] = pdata[metric]

        # config
        if 'config' in ftm:
            record.update(_extract_config_fields(ftm['config']))

    # ---- 2) metrics.json ----
    met = _load_json(os.path.join(exp_dir, 'metrics.json'))
    if met:
        sources.append('metrics')
        if _is_missing(record.get('exp_name')):
            record['exp_name'] = met.get('exp_name', exp_name)

        # 从 metrics.json 填充 (仅在尚未有值时)
        # 键名映射: metrics.json 中的字段名 → 汇总表字段名
        field_map = {
            'accuracy': 'test_acc',
            'test_accuracy': 'test_acc',
            'test_mae': 'test_mae',
            'test_rmse': 'test_rmse',
            'test_loss': 'test_loss',
            'zero_baseline_mae': 'zero_baseline_mae',
            'zero_baseline_rmse': 'zero_baseline_rmse',
            'avg_spike_rate': 'test_spike_rate',
            'sparsity': 'test_sparsity',
            'spikes_per_image': 'test_spikes_per_image',
            'test_samples': 'test_samples',
            'total_test_frames': 'test_samples',
            'epoch': 'best_epoch',
        }
        for src_key, dst_key in field_map.items():
            if src_key in met and _is_missing(record.get(dst_key)):
                record[dst_key] = met[src_key]

        # 直接同名字段 (config 类信息)
        direct_fields = [
            'neuron_type', 'residual_mode', 'T', 'dataset',
            'task_name', 'img_h', 'img_w', 'seq_len', 'stride',
            'mode', 'encoding', 'task_num_classes',
        ]
        for f in direct_fields:
            if f in met and _is_missing(record.get(f)):
                record[f] = met[f]

    # ---- 3) best_model.ckpt ----
    ckpt_path = os.path.join(exp_dir, 'best_model.ckpt')
    if not os.path.isfile(ckpt_path):
        # 尝试其他常见命名
        for alt in ('best.ckpt', 'best_model.pth', 'best.pth'):
            alt_path = os.path.join(exp_dir, alt)
            if os.path.isfile(alt_path):
                ckpt_path = alt_path
                break

    ckpt_meta = None
    if os.path.isfile(ckpt_path):
        ckpt_meta = _load_ckpt_meta(ckpt_path)
        if ckpt_meta:
            sources.append('ckpt')
            # 从 checkpoint 补充
            if 'max_test_acc' in ckpt_meta and _is_missing(record.get('best_val_acc')):
                record['best_val_acc'] = ckpt_meta['max_test_acc']
            if 'max_val_acc' in ckpt_meta and _is_missing(record.get('best_val_acc')):
                record['best_val_acc'] = ckpt_meta['max_val_acc']
            if 'min_val_mae' in ckpt_meta and _is_missing(record.get('best_val_mae')):
                record['best_val_mae'] = ckpt_meta['min_val_mae']
            if 'epoch' in ckpt_meta and _is_missing(record.get('best_epoch')):
                record['best_epoch'] = ckpt_meta['epoch']
            if 'config' in ckpt_meta:
                cfg_fields = _extract_config_fields(ckpt_meta['config'])
                for k, v in cfg_fields.items():
                    if _is_missing(record.get(k)):
                        record[k] = v

    # ---- 4) pipeline_log.json (补充信息) ----
    plog = _load_json(os.path.join(exp_dir, 'pipeline_log.json'))
    if plog:
        if 'task_type' in plog and _is_missing(record.get('task_name')):
            record['task_name'] = plog['task_type']

    # ---- 5) metadata fallback（增强 results 汇总鲁棒性）----
    # 5.1 test_samples 兜底：兼容 plot_results.py 输出 total_test_frames
    if _is_missing(record.get('test_samples')):
        if met and not _is_missing(met.get('test_samples')):
            record['test_samples'] = met.get('test_samples')
        elif met and not _is_missing(met.get('total_test_frames')):
            record['test_samples'] = met.get('total_test_frames')
        elif not _is_missing(record.get('total_test_frames')):
            record['test_samples'] = record.get('total_test_frames')

    # 5.2 当 dataset / task_name 缺失时，优先再尝试 ckpt config
    if (_is_missing(record.get('dataset')) or _is_missing(record.get('task_name'))):
        if ckpt_meta and isinstance(ckpt_meta, dict) and isinstance(ckpt_meta.get('config'), dict):
            cfg_fields = _extract_config_fields(ckpt_meta['config'])
            if not _is_missing(cfg_fields.get('dataset')) and _is_missing(record.get('dataset')):
                record['dataset'] = cfg_fields['dataset']
            if not _is_missing(cfg_fields.get('task_name')) and _is_missing(record.get('task_name')):
                record['task_name'] = cfg_fields['task_name']

    # 5.3 启发式推断：从 exp_name / 目录名补齐缺失 metadata
    # 注意：只在缺失字段时补齐，不覆盖已有值
    name_candidates = [
        record.get('exp_name'),
        exp_name,
        os.path.basename(exp_dir),
    ]
    for name in name_candidates:
        hint = _infer_meta_from_name(name)
        if _is_missing(record.get('task_name')) and not _is_missing(hint.get('task_name')):
            record['task_name'] = hint['task_name']
        if _is_missing(record.get('dataset')) and not _is_missing(hint.get('dataset')):
            record['dataset'] = hint['dataset']

    # 5.4 task_name 缺失但 exp_name 中包含明确 token 时，自动补齐
    #（与上一步互补，确保 stage3/stage4/junction_lr/action3_balanced 可识别）
    if _is_missing(record.get('task_name')):
        token_hint = _infer_meta_from_name(record.get('exp_name', exp_name))
        if not _is_missing(token_hint.get('task_name')):
            record['task_name'] = token_hint['task_name']

    # 5.5 dataset 缺失但 task_name 属于 corridor_task 系列，自动补 corridor_task
    if _is_missing(record.get('dataset')):
        task_l = str(record.get('task_name', '')).lower()
        if task_l in ('stage3', 'stage4', 'junction_lr', 'action3_balanced'):
            record['dataset'] = 'corridor_task'

    # 没有任何数据来源则返回 None
    if not sources:
        return None

    record['source'] = '+'.join(sources)

    # ---- 计算 zero baseline 增益百分比 ----
    t_mae = record.get('test_mae')
    z_mae = record.get('zero_baseline_mae')
    if t_mae is not None and z_mae is not None:
        try:
            z_mae_f = float(z_mae)
            if z_mae_f > 0:
                record['mae_gain_vs_zero_pct'] = round(
                    (z_mae_f - float(t_mae)) / z_mae_f * 100, 2)
        except (ValueError, TypeError):
            pass

    t_rmse = record.get('test_rmse')
    z_rmse = record.get('zero_baseline_rmse')
    if t_rmse is not None and z_rmse is not None:
        try:
            z_rmse_f = float(z_rmse)
            if z_rmse_f > 0:
                record['rmse_gain_vs_zero_pct'] = round(
                    (z_rmse_f - float(t_rmse)) / z_rmse_f * 100, 2)
        except (ValueError, TypeError):
            pass

    return record


# ============================================================================
# 扫描 & 汇总
# ============================================================================

def scan_experiments(exp_root, recursive=False):
    """
    扫描实验目录。

    Args:
        exp_root: 实验根目录
        recursive: 是否递归扫描子目录

    Returns:
        list of dict
    """
    records = []
    scanned = 0
    skipped = 0

    if not os.path.isdir(exp_root):
        print(f'  ✗ 实验目录不存在: {exp_root}')
        return records

    candidates = []

    if recursive:
        for root, dirs, files in os.walk(exp_root):
            # 检查当前目录是否是实验目录
            has_result = any(
                f in files for f in (
                    'final_test_metrics.json', 'metrics.json',
                    'best_model.ckpt', 'best.ckpt',
                    'best_model.pth', 'best.pth'))
            if has_result:
                candidates.append(root)
    else:
        for name in sorted(os.listdir(exp_root)):
            d = os.path.join(exp_root, name)
            if os.path.isdir(d):
                candidates.append(d)

    for exp_dir in sorted(candidates):
        scanned += 1
        record = read_experiment(exp_dir)
        if record:
            records.append(record)
        else:
            skipped += 1

    print(f'  扫描: {scanned} 目录, 有效: {len(records)}, 跳过: {skipped}')
    return records


def sort_records(records, sort_by=None, ascending=None):
    """
    按指定字段排序。

    自动检测: 若 sort_by 含 'mae' / 'loss' 则升序，否则降序。
    """
    if not records or not sort_by:
        return records

    if ascending is None:
        ascending = any(kw in sort_by.lower()
                        for kw in ('mae', 'loss'))

    def _key(r):
        v = r.get(sort_by)
        if v is None:
            return float('inf') if ascending else float('-inf')
        try:
            return float(v)
        except (ValueError, TypeError):
            return float('inf') if ascending else float('-inf')

    return sorted(records, key=_key, reverse=not ascending)


# ============================================================================
# 输出
# ============================================================================

def _fmt(val, field):
    """格式化单个值"""
    if val is None or val == '':
        return ''
    fmt = FIELD_FMT.get(field)
    if fmt:
        try:
            return f'{float(val):{fmt}}'
        except (ValueError, TypeError):
            return str(val)
    return str(val)


def write_csv(out_csv, records):
    """写出 CSV"""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    # 收集所有出现的字段
    all_fields = []
    seen = set()
    for f in SUMMARY_FIELDS:
        if f not in seen:
            all_fields.append(f)
            seen.add(f)
    for r in records:
        for k in r.keys():
            if k not in seen:
                all_fields.append(k)
                seen.add(k)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    return all_fields


def write_markdown(out_md, records, is_regression=False):
    """写出 Markdown 论文表格"""
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)

    fields = MD_FIELDS_REG if is_regression else MD_FIELDS_CLS

    # 过滤掉没有数据的列
    active = [f for f in fields
              if any(r.get(f) is not None for r in records)]

    # 表头映射
    header_map = {
        'exp_name': 'Experiment',
        'neuron_type': 'Neuron',
        'residual_mode': 'Residual',
        'T': 'T',
        'best_val_acc': 'Val Acc',
        'best_val_mae': 'Val MAE',
        'test_acc': 'Test Acc',
        'test_mae': 'Test MAE',
        'test_rmse': 'Test RMSE',
        'zero_baseline_mae': 'Zero MAE',
        'zero_baseline_rmse': 'Zero RMSE',
        'mae_gain_vs_zero_pct': 'MAE Gain%',
        'rmse_gain_vs_zero_pct': 'RMSE Gain%',
        'correcting_mae': 'Corr MAE',
        'correcting_rmse': 'Corr RMSE',
        'settled_mae': 'Settl MAE',
        'settled_rmse': 'Settl RMSE',
        'test_loss': 'Test Loss',
        'test_spike_rate': 'Spike Rate',
        'test_sparsity': 'Sparsity',
        'test_spikes_per_image': 'Spk/Img',
    }

    lines = []
    lines.append(f'# 实验结果汇总\n')
    lines.append(f'> {len(records)} 个实验\n')
    lines.append('')

    # 表头
    headers = [header_map.get(f, f) for f in active]
    lines.append('| ' + ' | '.join(headers) + ' |')
    # 对齐: 数值列右对齐
    aligns = []
    for f in active:
        if f in FIELD_FMT or f == 'T':
            aligns.append('---:')
        else:
            aligns.append(':---')
    lines.append('| ' + ' | '.join(aligns) + ' |')

    # 数据行
    for r in records:
        row = [_fmt(r.get(f), f) for f in active]
        lines.append('| ' + ' | '.join(row) + ' |')

    lines.append('')

    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def print_top_k(records, sort_field, k=5, ascending=False):
    """打印 Top-K 实验"""
    sorted_recs = sort_records(records, sort_by=sort_field,
                               ascending=ascending)[:k]
    if not sorted_recs:
        return

    print(f'\n  ── Top-{k} (按 {sort_field}, '
          f'{"升序" if ascending else "降序"}) ──')
    for i, r in enumerate(sorted_recs, 1):
        val = r.get(sort_field)
        val_str = _fmt(val, sort_field) if val is not None else 'N/A'
        name = r.get('exp_name', '?')
        neuron = r.get('neuron_type', '?')
        T = r.get('T', '?')
        print(f'    #{i:2d}  {name:40s}  {sort_field}={val_str}'
              f'  neuron={neuron} T={T}')


# ============================================================================
# 主入口
# ============================================================================

def collect_results(exp_root, out_csv=None, out_md=None,
                    recursive=False, prefer_final_test=True,
                    sort_by=None, top_k=5):
    """
    汇总实验结果核心函数。

    Returns:
        list of dict: 汇总记录
    """
    print(f'{"=" * 60}')
    print(f'  实验结果自动汇总')
    print(f'{"=" * 60}')
    print(f'  实验目录:   {os.path.abspath(exp_root)}')
    print(f'  递归扫描:   {recursive}')
    if out_csv:
        print(f'  CSV 输出:   {out_csv}')
    if out_md:
        print(f'  MD 输出:    {out_md}')
    print(f'{"=" * 60}')

    # ---- 扫描 ----
    print(f'\n[1/3] 扫描实验目录...')
    records = scan_experiments(exp_root, recursive=recursive)

    if not records:
        print(f'  ✗ 未找到任何有效实验结果!')
        return records

    # ---- 判断是否含有回归任务 ----
    has_regression = any(r.get('test_mae') is not None for r in records)
    has_classification = any(r.get('test_acc') is not None for r in records)

    # 自动确定排序字段
    if sort_by is None:
        if has_classification:
            sort_by = 'test_acc'
        elif has_regression:
            sort_by = 'test_mae'
        else:
            sort_by = 'exp_name'

    # 排序
    ascending = any(kw in sort_by.lower()
                    for kw in ('mae', 'loss'))
    records = sort_records(records, sort_by=sort_by, ascending=ascending)

    # ---- 输出 ----
    print(f'\n[2/3] 生成汇总文件...')

    if out_csv:
        fields = write_csv(out_csv, records)
        print(f'  [✓] {out_csv} ({len(fields)} 列)')

    if out_md:
        write_markdown(out_md, records, is_regression=has_regression)
        print(f'  [✓] {out_md}')

    # ---- 统计 ----
    print(f'\n[3/3] 统计...')
    print(f'{"=" * 60}')
    print(f'  总实验数: {len(records)}')

    # 按数据来源统计
    src_dist = defaultdict(int)
    for r in records:
        src_dist[r.get('source', '?')] += 1
    print(f'\n  ── 数据来源 ──')
    for src, cnt in sorted(src_dist.items()):
        print(f'    {src:25s}: {cnt}')

    # 按 neuron_type 统计
    neuron_dist = defaultdict(int)
    for r in records:
        nt = r.get('neuron_type', '(unknown)')
        neuron_dist[nt] += 1
    if len(neuron_dist) > 1 or '(unknown)' not in neuron_dist:
        print(f'\n  ── Neuron 分布 ──')
        for nt, cnt in sorted(neuron_dist.items()):
            print(f'    {nt:25s}: {cnt}')

    # 按 task_name 统计
    task_dist = defaultdict(int)
    for r in records:
        tn = r.get('task_name', r.get('dataset', '(unknown)'))
        task_dist[tn] += 1
    if len(task_dist) > 1 or '(unknown)' not in task_dist:
        print(f'\n  ── Task 分布 ──')
        for tn, cnt in sorted(task_dist.items()):
            print(f'    {tn:25s}: {cnt}')

    # Top-K
    if has_classification and top_k > 0:
        print_top_k(records, 'test_acc', k=top_k, ascending=False)
    if has_regression and top_k > 0:
        print_top_k(records, 'test_mae', k=top_k, ascending=True)
        # RMSE Top-K (若有)
        has_rmse = any(r.get('test_rmse') is not None for r in records)
        if has_rmse:
            print_top_k(records, 'test_rmse', k=top_k, ascending=True)

    print(f'\n{"=" * 60}')
    return records


def main():
    parser = argparse.ArgumentParser(
        description='实验结果自动汇总 — 生成 CSV/Markdown 消融实验表格',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp_root', type=str,
                        default='./results',
                        help='实验结果根目录')
    parser.add_argument('--out_csv', type=str,
                        default=None,
                        help='CSV 输出路径 (默认 exp_root/ablation_summary.csv)')
    parser.add_argument('--out_md', type=str,
                        default=None,
                        help='Markdown 输出路径 '
                             '(默认 exp_root/ablation_summary.md)')
    parser.add_argument('--recursive', action='store_true',
                        help='递归扫描子目录')
    parser.add_argument('--prefer_final_test', type=lambda x:
                        str(x).lower() in ('true', '1', 'yes'),
                        default=True,
                        help='优先使用 final_test_metrics.json')
    parser.add_argument('--sort_by', type=str, default=None,
                        help='排序字段 (默认: 分类按 test_acc 降序, '
                             '回归按 test_mae 升序)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='打印 Top-K 实验')

    args = parser.parse_args()

    # 默认输出路径
    if args.out_csv is None:
        args.out_csv = os.path.join(args.exp_root,
                                    'ablation_summary.csv')
    if args.out_md is None:
        args.out_md = os.path.join(args.exp_root,
                                   'ablation_summary.md')

    collect_results(
        exp_root=args.exp_root,
        out_csv=args.out_csv,
        out_md=args.out_md,
        recursive=args.recursive,
        prefer_final_test=args.prefer_final_test,
        sort_by=args.sort_by,
        top_k=args.top_k,
    )


if __name__ == '__main__':
    main()
