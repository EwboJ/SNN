"""
代理功耗分析汇总脚本（无实物部署阶段）
=================================================
从 ablation_summary.csv 读取实验结果，生成：
  1) proxy_power_summary_cls.csv
  2) proxy_power_summary_reg.csv
  3) proxy_power_summary.md

用法:
    python scripts/summarize_proxy_power.py \
        --csv_path results/ablation_summary.csv \
        --out_dir results
"""

import os
import csv
import math
import argparse
from collections import OrderedDict


# ============================================================================
# 常量
# ============================================================================

BASE_META_FIELDS = [
    'exp_name',
    'dataset',
    'task_name',
    'neuron_type',
    'residual_mode',
    'T',
    'source',
]

CLS_FIELDS = BASE_META_FIELDS + [
    'test_acc',
    'test_spike_rate',
    'test_spikes_per_image',
    'test_sparsity',
    'proxy_activity_index',
    'sparsity',
    'performance_activity_ratio',
    'performance_spike_rate_ratio',
]

REG_FIELDS = BASE_META_FIELDS + [
    'test_mae',
    'test_rmse',
    'test_spike_rate',
    'test_spikes_per_image',
    'test_sparsity',
    'correcting_mae',
    'settled_mae',
    'proxy_activity_index',
    'sparsity',
    'mae_activity_product',
    'rmse_activity_product',
    'inverse_mae_activity_ratio',
]

FIELD_FMT = {
    'test_acc': '.4f',
    'test_mae': '.4f',
    'test_rmse': '.4f',
    'test_spike_rate': '.6f',
    'test_spikes_per_image': '.1f',
    'test_sparsity': '.6f',
    'proxy_activity_index': '.1f',
    'sparsity': '.6f',
    'performance_activity_ratio': '.8f',
    'performance_spike_rate_ratio': '.8f',
    'mae_activity_product': '.8f',
    'rmse_activity_product': '.8f',
    'inverse_mae_activity_ratio': '.8f',
    'correcting_mae': '.4f',
    'settled_mae': '.4f',
}


# ============================================================================
# 读取与计算辅助
# ============================================================================

def _is_missing(v):
    return v is None or (isinstance(v, str) and v.strip() == '')


def _to_float(v, default=None):
    if _is_missing(v):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _safe_div(numer, denom, eps=0.0):
    n = _to_float(numer, None)
    d = _to_float(denom, None)
    if n is None or d is None:
        return None
    d_eff = d + float(eps)
    if abs(d_eff) <= 1e-12:
        return None
    return n / d_eff


def _load_csv(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'未找到输入 CSV: {csv_path}')
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _sort_records(records, key, ascending=False):
    if not records:
        return records

    def _key(r):
        v = _to_float(r.get(key), None)
        if v is None:
            return float('inf') if ascending else float('-inf')
        return v

    return sorted(records, key=_key, reverse=not ascending)


def _build_cls_record(row):
    """
    分类任务记录（存在 test_acc）。
    """
    test_acc = _to_float(row.get('test_acc'), None)
    if test_acc is None:
        return None

    test_spike_rate = _to_float(row.get('test_spike_rate'), None)
    test_spikes_per_image = _to_float(row.get('test_spikes_per_image'), None)
    test_sparsity = _to_float(row.get('test_sparsity'), None)

    rec = OrderedDict()
    for f in BASE_META_FIELDS:
        rec[f] = row.get(f, '')

    rec['test_acc'] = test_acc
    rec['test_spike_rate'] = test_spike_rate
    rec['test_spikes_per_image'] = test_spikes_per_image
    rec['test_sparsity'] = test_sparsity

    # 代理功耗相关指标（命名明确为 proxy/activity）
    rec['proxy_activity_index'] = test_spikes_per_image
    rec['sparsity'] = test_sparsity
    rec['performance_activity_ratio'] = _safe_div(
        test_acc, test_spikes_per_image
    )
    rec['performance_spike_rate_ratio'] = _safe_div(
        test_acc, test_spike_rate
    )
    return rec


def _build_reg_record(row):
    """
    回归任务记录（存在 test_mae）。
    """
    test_mae = _to_float(row.get('test_mae'), None)
    if test_mae is None:
        return None

    test_rmse = _to_float(row.get('test_rmse'), None)
    test_spike_rate = _to_float(row.get('test_spike_rate'), None)
    test_spikes_per_image = _to_float(row.get('test_spikes_per_image'), None)
    test_sparsity = _to_float(row.get('test_sparsity'), None)
    correcting_mae = _to_float(row.get('correcting_mae'), None)
    settled_mae = _to_float(row.get('settled_mae'), None)

    rec = OrderedDict()
    for f in BASE_META_FIELDS:
        rec[f] = row.get(f, '')

    rec['test_mae'] = test_mae
    rec['test_rmse'] = test_rmse
    rec['test_spike_rate'] = test_spike_rate
    rec['test_spikes_per_image'] = test_spikes_per_image
    rec['test_sparsity'] = test_sparsity
    rec['correcting_mae'] = correcting_mae
    rec['settled_mae'] = settled_mae

    # 代理功耗相关指标（命名明确为 proxy/activity）
    rec['proxy_activity_index'] = test_spikes_per_image
    rec['sparsity'] = test_sparsity

    mae_activity_product = None
    if test_mae is not None and test_spikes_per_image is not None:
        mae_activity_product = test_mae * test_spikes_per_image
    rec['mae_activity_product'] = mae_activity_product

    rmse_activity_product = None
    if test_rmse is not None and test_spikes_per_image is not None:
        rmse_activity_product = test_rmse * test_spikes_per_image
    rec['rmse_activity_product'] = rmse_activity_product

    if mae_activity_product is None:
        rec['inverse_mae_activity_ratio'] = None
    else:
        rec['inverse_mae_activity_ratio'] = 1.0 / (mae_activity_product + 1e-8)

    return rec


# ============================================================================
# 输出
# ============================================================================

def _fmt(v, field):
    if _is_missing(v):
        return ''
    fmt = FIELD_FMT.get(field)
    if fmt:
        try:
            return f'{float(v):{fmt}}'
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def _write_csv(path, records, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in records:
            writer.writerow({k: _fmt(r.get(k), k) for k in fields})


def _write_md(path, cls_records, reg_records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    cls_headers = [
        'exp_name', 'neuron_type', 'residual_mode', 'T',
        'test_acc', 'proxy_activity_index',
        'performance_activity_ratio', 'performance_spike_rate_ratio',
        'sparsity',
    ]
    reg_headers = [
        'exp_name', 'neuron_type', 'residual_mode', 'T',
        'test_mae', 'test_rmse', 'proxy_activity_index',
        'mae_activity_product', 'rmse_activity_product',
        'inverse_mae_activity_ratio', 'correcting_mae', 'settled_mae',
        'sparsity',
    ]

    lines = []
    lines.append('# 代理功耗分析汇总（无实物部署阶段）')
    lines.append('')
    lines.append(f'- 分类记录数: {len(cls_records)}')
    lines.append(f'- 回归记录数: {len(reg_records)}')
    lines.append('')

    lines.append('## 分类任务（按 performance_activity_ratio 降序）')
    lines.append('')
    if cls_records:
        lines.append('| ' + ' | '.join(cls_headers) + ' |')
        lines.append('| ' + ' | '.join(['---' if h == 'exp_name' else '---:' for h in cls_headers]) + ' |')
        for r in cls_records:
            row = [_fmt(r.get(h), h) for h in cls_headers]
            lines.append('| ' + ' | '.join(row) + ' |')
    else:
        lines.append('无分类任务记录。')
    lines.append('')

    lines.append('## 回归任务（按 mae_activity_product 升序）')
    lines.append('')
    if reg_records:
        lines.append('| ' + ' | '.join(reg_headers) + ' |')
        lines.append('| ' + ' | '.join(['---' if h == 'exp_name' else '---:' for h in reg_headers]) + ' |')
        for r in reg_records:
            row = [_fmt(r.get(h), h) for h in reg_headers]
            lines.append('| ' + ' | '.join(row) + ' |')
    else:
        lines.append('无回归任务记录。')
    lines.append('')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================================================
# 主流程
# ============================================================================

def summarize_proxy_power(csv_path, out_dir):
    print('=' * 64)
    print('  无实物部署阶段代理功耗分析汇总')
    print('=' * 64)
    print(f'  输入 CSV: {os.path.abspath(csv_path)}')
    print(f'  输出目录: {os.path.abspath(out_dir)}')
    print('=' * 64)

    rows = _load_csv(csv_path)
    print(f'\n[1/3] 读取输入 CSV ... 共 {len(rows)} 行')

    cls_records = []
    reg_records = []
    for row in rows:
        cls_rec = _build_cls_record(row)
        if cls_rec is not None:
            cls_records.append(cls_rec)

        reg_rec = _build_reg_record(row)
        if reg_rec is not None:
            reg_records.append(reg_rec)

    # 排序规则：
    # 分类：performance_activity_ratio 降序
    # 回归：mae_activity_product 升序
    cls_records = _sort_records(
        cls_records, key='performance_activity_ratio', ascending=False)
    reg_records = _sort_records(
        reg_records, key='mae_activity_product', ascending=True)

    print(f'  分类记录: {len(cls_records)}')
    print(f'  回归记录: {len(reg_records)}')

    print('\n[2/3] 写出 CSV ...')
    cls_csv = os.path.join(out_dir, 'proxy_power_summary_cls.csv')
    reg_csv = os.path.join(out_dir, 'proxy_power_summary_reg.csv')
    _write_csv(cls_csv, cls_records, CLS_FIELDS)
    _write_csv(reg_csv, reg_records, REG_FIELDS)
    print(f'  [✓] {cls_csv}')
    print(f'  [✓] {reg_csv}')

    print('\n[3/3] 写出 Markdown ...')
    md_path = os.path.join(out_dir, 'proxy_power_summary.md')
    _write_md(md_path, cls_records, reg_records)
    print(f'  [✓] {md_path}')

    print('\n' + '=' * 64)
    print('  汇总完成')
    print('=' * 64)


def main():
    parser = argparse.ArgumentParser(
        description='无实物部署阶段代理功耗分析汇总',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--csv_path', type=str, required=True,
        help='ablation_summary.csv 路径')
    parser.add_argument(
        '--out_dir', type=str, required=True,
        help='输出目录（将写出 cls/reg CSV + Markdown）')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summarize_proxy_power(csv_path=args.csv_path, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
