"""
Summarize multi-task proxy activity analysis from ablation_summary.csv.

Outputs:
  1) proxy_power_summary_cls.csv
  2) proxy_power_summary_reg.csv
  3) proxy_power_summary.md
  4) proxy_power_summary_by_task.md
"""

import argparse
import csv
import math
import os
from collections import OrderedDict


CLS_OPTIONAL_IF_PRESENT = ["macro_f1"]
REG_OPTIONAL_IF_PRESENT = ["correcting_mae", "settled_mae"]

CLS_FIELDS_BASE = [
    "task_name",
    "exp_name",
    "test_acc",
    "test_spike_rate",
    "test_sparsity",
    "test_spikes_per_image",
    "proxy_activity_index",
    "performance_activity_ratio",
    "performance_spike_rate_ratio",
]

REG_FIELDS_BASE = [
    "task_name",
    "exp_name",
    "test_mae",
    "test_rmse",
    "zero_baseline_mae",
    "zero_baseline_rmse",
    "mae_gain_vs_zero_pct",
    "rmse_gain_vs_zero_pct",
    "test_spike_rate",
    "test_sparsity",
    "test_spikes_per_image",
    "mae_activity_product",
    "rmse_activity_product",
    "inverse_mae_activity_ratio",
]

FIELD_FMT = {
    "test_acc": ".4f",
    "macro_f1": ".4f",
    "test_mae": ".4f",
    "test_rmse": ".4f",
    "zero_baseline_mae": ".4f",
    "zero_baseline_rmse": ".4f",
    "mae_gain_vs_zero_pct": ".2f",
    "rmse_gain_vs_zero_pct": ".2f",
    "test_spike_rate": ".6f",
    "test_spikes_per_image": ".1f",
    "test_sparsity": ".6f",
    "proxy_activity_index": ".1f",
    "performance_activity_ratio": ".8f",
    "performance_spike_rate_ratio": ".8f",
    "mae_activity_product": ".8f",
    "rmse_activity_product": ".8f",
    "inverse_mae_activity_ratio": ".8f",
    "correcting_mae": ".4f",
    "settled_mae": ".4f",
}


def _is_missing(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def _to_float(value, default=None):
    if _is_missing(value):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


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
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))
    return rows, fieldnames


def _build_cls_fields(csv_columns):
    fields = list(CLS_FIELDS_BASE)
    if "macro_f1" in csv_columns:
        insert_at = fields.index("test_acc") + 1
        fields.insert(insert_at, "macro_f1")
    return fields


def _build_reg_fields(csv_columns):
    fields = list(REG_FIELDS_BASE)
    for opt in REG_OPTIONAL_IF_PRESENT:
        if opt in csv_columns:
            fields.append(opt)
    return fields


def _sort_records(records, key, ascending=False):
    if not records:
        return records

    def _key(record):
        value = _to_float(record.get(key), None)
        if value is None:
            return float("inf") if ascending else float("-inf")
        return value

    return sorted(records, key=_key, reverse=not ascending)


def _build_cls_record(row):
    test_acc = _to_float(row.get("test_acc"), None)
    if test_acc is None:
        return None

    test_spike_rate = _to_float(row.get("test_spike_rate"), None)
    test_sparsity = _to_float(row.get("test_sparsity"), None)
    test_spikes_per_image = _to_float(row.get("test_spikes_per_image"), None)

    record = OrderedDict()
    record["task_name"] = row.get("task_name", "")
    record["exp_name"] = row.get("exp_name", "")
    record["test_acc"] = test_acc
    record["macro_f1"] = _to_float(row.get("macro_f1"), None)
    record["test_spike_rate"] = test_spike_rate
    record["test_sparsity"] = test_sparsity
    record["test_spikes_per_image"] = test_spikes_per_image
    record["proxy_activity_index"] = test_spikes_per_image
    record["performance_activity_ratio"] = _safe_div(test_acc, test_spikes_per_image)
    record["performance_spike_rate_ratio"] = _safe_div(test_acc, test_spike_rate)
    return record


def _derive_gain_vs_zero_pct(test_metric, zero_baseline_metric):
    if test_metric is None or zero_baseline_metric is None:
        return None
    if abs(zero_baseline_metric) <= 1e-12:
        return None
    return (zero_baseline_metric - test_metric) / zero_baseline_metric * 100.0


def _build_reg_record(row):
    test_mae = _to_float(row.get("test_mae"), None)
    if test_mae is None:
        return None

    test_rmse = _to_float(row.get("test_rmse"), None)
    zero_baseline_mae = _to_float(row.get("zero_baseline_mae"), None)
    zero_baseline_rmse = _to_float(row.get("zero_baseline_rmse"), None)

    mae_gain_vs_zero_pct = _to_float(row.get("mae_gain_vs_zero_pct"), None)
    if mae_gain_vs_zero_pct is None:
        mae_gain_vs_zero_pct = _derive_gain_vs_zero_pct(test_mae, zero_baseline_mae)

    rmse_gain_vs_zero_pct = _to_float(row.get("rmse_gain_vs_zero_pct"), None)
    if rmse_gain_vs_zero_pct is None:
        rmse_gain_vs_zero_pct = _derive_gain_vs_zero_pct(test_rmse, zero_baseline_rmse)

    test_spike_rate = _to_float(row.get("test_spike_rate"), None)
    test_sparsity = _to_float(row.get("test_sparsity"), None)
    test_spikes_per_image = _to_float(row.get("test_spikes_per_image"), None)

    mae_activity_product = None
    if test_mae is not None and test_spikes_per_image is not None:
        mae_activity_product = test_mae * test_spikes_per_image

    rmse_activity_product = None
    if test_rmse is not None and test_spikes_per_image is not None:
        rmse_activity_product = test_rmse * test_spikes_per_image

    inverse_mae_activity_ratio = None
    if mae_activity_product is not None:
        inverse_mae_activity_ratio = 1.0 / (mae_activity_product + 1e-8)

    record = OrderedDict()
    record["task_name"] = row.get("task_name", "")
    record["exp_name"] = row.get("exp_name", "")
    record["test_mae"] = test_mae
    record["test_rmse"] = test_rmse
    record["zero_baseline_mae"] = zero_baseline_mae
    record["zero_baseline_rmse"] = zero_baseline_rmse
    record["mae_gain_vs_zero_pct"] = mae_gain_vs_zero_pct
    record["rmse_gain_vs_zero_pct"] = rmse_gain_vs_zero_pct
    record["test_spike_rate"] = test_spike_rate
    record["test_sparsity"] = test_sparsity
    record["test_spikes_per_image"] = test_spikes_per_image
    record["mae_activity_product"] = mae_activity_product
    record["rmse_activity_product"] = rmse_activity_product
    record["inverse_mae_activity_ratio"] = inverse_mae_activity_ratio
    record["correcting_mae"] = _to_float(row.get("correcting_mae"), None)
    record["settled_mae"] = _to_float(row.get("settled_mae"), None)
    return record


def _fmt(value, field):
    if _is_missing(value):
        return ""
    precision = FIELD_FMT.get(field)
    if precision:
        try:
            return f"{float(value):{precision}}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _write_csv(path, records, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({name: _fmt(record.get(name), name) for name in fields})


def _md_align_row(headers):
    return ["---" if h in ("task_name", "exp_name") else "---:" for h in headers]


def _append_md_table(lines, records, headers):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(_md_align_row(headers)) + " |")
    for record in records:
        row = [_fmt(record.get(header), header) for header in headers]
        lines.append("| " + " | ".join(row) + " |")


def _write_md(path, cls_records, reg_records, cls_fields, reg_fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    lines = []
    lines.append("# Proxy Activity Summary")
    lines.append("")
    lines.append(f"- classification_records: {len(cls_records)}")
    lines.append(f"- regression_records: {len(reg_records)}")
    lines.append("")

    lines.append("## Classification (sorted by performance_activity_ratio desc)")
    lines.append("")
    if cls_records:
        _append_md_table(lines, cls_records, cls_fields)
    else:
        lines.append("No classification records.")
    lines.append("")

    lines.append("## Regression (sorted by mae_activity_product asc)")
    lines.append("")
    if reg_records:
        _append_md_table(lines, reg_records, reg_fields)
    else:
        lines.append("No regression records.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _group_by_task(records):
    grouped = OrderedDict()
    for record in records:
        task_name = str(record.get("task_name", "")).strip() or "unknown_task"
        if task_name not in grouped:
            grouped[task_name] = []
        grouped[task_name].append(record)
    return grouped


def _write_md_by_task(path, cls_records, reg_records, cls_fields, reg_fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    cls_grouped = _group_by_task(cls_records)
    reg_grouped = _group_by_task(reg_records)
    task_names = sorted(set(cls_grouped.keys()) | set(reg_grouped.keys()))

    lines = []
    lines.append("# Proxy Activity Summary By Task")
    lines.append("")
    lines.append(
        "Classification tables are sorted by performance_activity_ratio desc."
    )
    lines.append("Regression tables are sorted by mae_activity_product asc.")
    lines.append("")

    if not task_names:
        lines.append("No records.")
    else:
        for task_name in task_names:
            lines.append(f"## {task_name}")
            lines.append("")

            task_cls_records = cls_grouped.get(task_name, [])
            lines.append("### Classification")
            lines.append("")
            if task_cls_records:
                _append_md_table(lines, task_cls_records, cls_fields)
            else:
                lines.append("No classification records.")
            lines.append("")

            task_reg_records = reg_grouped.get(task_name, [])
            lines.append("### Regression")
            lines.append("")
            if task_reg_records:
                _append_md_table(lines, task_reg_records, reg_fields)
            else:
                lines.append("No regression records.")
            lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def summarize_proxy_power(csv_path, out_dir):
    print("=" * 64)
    print("  Proxy activity summarization")
    print("=" * 64)
    print(f"  Input CSV: {os.path.abspath(csv_path)}")
    print(f"  Output dir: {os.path.abspath(out_dir)}")
    print("=" * 64)

    rows, csv_columns = _load_csv(csv_path)
    csv_column_set = set(csv_columns)
    print(f"\n[1/4] Read CSV ... total rows: {len(rows)}")

    cls_fields = _build_cls_fields(csv_column_set)
    reg_fields = _build_reg_fields(csv_column_set)

    cls_records = []
    reg_records = []
    for row in rows:
        cls_record = _build_cls_record(row)
        if cls_record is not None:
            cls_records.append(cls_record)

        reg_record = _build_reg_record(row)
        if reg_record is not None:
            reg_records.append(reg_record)

    cls_records = _sort_records(
        cls_records, key="performance_activity_ratio", ascending=False
    )
    reg_records = _sort_records(
        reg_records, key="mae_activity_product", ascending=True
    )

    print(f"  Classification records: {len(cls_records)}")
    print(f"  Regression records: {len(reg_records)}")

    print("\n[2/4] Write CSV ...")
    cls_csv = os.path.join(out_dir, "proxy_power_summary_cls.csv")
    reg_csv = os.path.join(out_dir, "proxy_power_summary_reg.csv")
    _write_csv(cls_csv, cls_records, cls_fields)
    _write_csv(reg_csv, reg_records, reg_fields)
    print(f"  [OK] {cls_csv}")
    print(f"  [OK] {reg_csv}")

    print("\n[3/4] Write overall Markdown ...")
    md_path = os.path.join(out_dir, "proxy_power_summary.md")
    _write_md(md_path, cls_records, reg_records, cls_fields, reg_fields)
    print(f"  [OK] {md_path}")

    print("\n[4/4] Write task-grouped Markdown ...")
    md_by_task_path = os.path.join(out_dir, "proxy_power_summary_by_task.md")
    _write_md_by_task(
        md_by_task_path, cls_records, reg_records, cls_fields, reg_fields
    )
    print(f"  [OK] {md_by_task_path}")

    print("\n" + "=" * 64)
    print("  Done")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize proxy activity metrics from ablation summary CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to ablation_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for cls/reg CSV and Markdown summaries",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summarize_proxy_power(csv_path=args.csv_path, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
