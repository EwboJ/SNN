#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 ROS2 /nav/debug 保存的 JSONL 日志。

使用说明:
  1. 推荐保存方式:
       ros2 topic echo /nav/debug --full-length --field data > center.jsonl

     也兼容未使用 --field data 时常见的单行格式:
       data: '{"state": "...", "tick_count": 1, ...}'

  2. 分析单个或多个日志:
       python3 scripts/analyze_nav_debug_logs.py \
         --logs center=path/to/center.jsonl left=path/to/left.jsonl right=path/to/right.jsonl

  3. 保存 CSV 汇总:
       python3 scripts/analyze_nav_debug_logs.py \
         --logs center=path/to/center.jsonl left=path/to/left.jsonl right=path/to/right.jsonl \
         --out_csv path/to/summary.csv

说明:
  - 只依赖 Python 标准库，不需要 ROS2 环境。
  - 空行和 "---" 分隔行会被忽略。
  - age/latency 统计会跳过 None/NaN/负数；在本项目调试字段里负数通常表示未知。
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Calibration = Dict[str, Any]

DIST_FIELDS = [
    "state",
    "reason",
    "safety_level",
    "omega_source",
    "pulse_recenter_state",
    "pulse_recenter_reason",
]

NUMERIC_FIELDS = [
    "cmd_linear_x",
    "cmd_angular_z",
    "straight_keep_raw_omega",
    "pulse_recenter_error",
    "image_age_ms",
    "straight_keep_age_ms",
    "straight_keep_latency_ms",
]

# 这些字段中负值一般代表未知值，不应进入分位数统计。
NONNEGATIVE_FIELDS = {
    "image_age_ms",
    "straight_keep_age_ms",
    "straight_keep_latency_ms",
}

HARD_STOP_REASON_NAMES = {
    "missing_image",
    "missing_image_or_timeout",
    "model_inference_exception",
    "model_submit_exception",
    "too_many_consecutive_errors",
    "exception",
    "control_tick_exception",
}

EPS = 1e-12
MISSING = object()


@dataclass
class LogSummary:
    label: str
    path: str
    raw_line_count: int = 0
    ignored_line_count: int = 0
    parse_error_count: int = 0
    total_rows: int = 0
    tick_values: List[int] = field(default_factory=list)
    counters: Dict[str, Counter] = field(
        default_factory=lambda: {name: Counter() for name in DIST_FIELDS}
    )
    values: Dict[str, List[float]] = field(
        default_factory=lambda: {name: [] for name in NUMERIC_FIELDS}
    )
    cmd_angular_valid_count: int = 0
    cmd_angular_nonzero_count: int = 0
    cmd_angular_positive_count: int = 0
    cmd_angular_negative_count: int = 0
    hard_stop_count: int = 0
    missing_image_or_timeout_count: int = 0
    model_output_timeout_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析 ROS2 /nav/debug JSONL 日志，输出导航、安全、omega 和时延统计。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python3 scripts/analyze_nav_debug_logs.py \\\n"
            "    --logs center=path/to/center.jsonl left=path/to/left.jsonl right=path/to/right.jsonl \\\n"
            "    --out_csv path/to/summary.csv"
        ),
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        metavar="LABEL=PATH",
        help="日志列表，建议使用 center=... left=... right=...；也可只传 PATH。",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="可选 CSV 输出路径。",
    )
    return parser.parse_args()


def parse_log_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    """解析 --logs 参数；无 label 时用文件名 stem 作为 label。"""
    parsed: List[Tuple[str, str]] = []
    seen = set()
    for index, spec in enumerate(specs, start=1):
        if "=" in spec:
            label, path = spec.split("=", 1)
            label = label.strip()
            path = path.strip()
        else:
            path = spec.strip()
            label = Path(path).stem or "log%d" % index

        if not label:
            raise ValueError("日志标签不能为空: %r" % spec)
        if not path:
            raise ValueError("日志路径不能为空: %r" % spec)
        if label in seen:
            raise ValueError("重复日志标签: %s" % label)
        seen.add(label)
        parsed.append((label, path))
    return parsed


def _load_structured_text(text: str) -> Any:
    """先按 JSON 解析，再兼容 ros2 echo 中单引号包住 JSON 字符串的情况。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return MISSING


def _coerce_debug_payload(obj: Any) -> Optional[Dict[str, Any]]:
    """把 JSON 字符串、std_msgs/String 包装等形式统一解成 debug dict。"""
    for _ in range(4):
        if isinstance(obj, dict):
            data = obj.get("data")
            if isinstance(data, str):
                inner = _load_structured_text(data.strip())
                if inner is not MISSING:
                    inner_payload = _coerce_debug_payload(inner)
                    if inner_payload is not None:
                        return inner_payload
            return obj

        if isinstance(obj, str):
            text = obj.strip()
            if not text or text == "---":
                return None
            inner = _load_structured_text(text)
            if inner is MISSING:
                return None
            obj = inner
            continue

        return None

    return None


def parse_debug_line(line: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """解析单行日志。

    返回 (payload, status)，status 为 ok / ignored / parse_error。
    """
    text = line.strip()
    if not text or text == "---":
        return None, "ignored"

    # ros2 topic echo std_msgs/String 时常见: data: '{"..."}'
    candidates = [text]
    if text.startswith("data:"):
        candidates.insert(0, text.split(":", 1)[1].strip())

    for candidate in candidates:
        obj = _load_structured_text(candidate)
        if obj is MISSING:
            continue
        payload = _coerce_debug_payload(obj)
        if payload is not None:
            return payload, "ok"

    return None, "parse_error"


def category_value(value: Any) -> str:
    if value is None:
        return "(missing)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value if value else "(empty)"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def safe_float(value: Any, *, nonnegative: bool = False) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None

    if not math.isfinite(number):
        return None
    if nonnegative and number < 0:
        return None
    return number


def safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                number = float(text)
            except ValueError:
                return None
            if not math.isfinite(number):
                return None
            return int(number)
    return None


def is_model_output_timeout(text: Any) -> bool:
    return str(text or "").startswith("model_output_timeout")


def is_hard_stop_payload(payload: Dict[str, Any]) -> bool:
    safety_level = str(payload.get("safety_level", "") or "").upper()
    if safety_level == "HARD_STOP":
        return True

    reason = str(payload.get("reason", "") or "")
    hard_stop_reason = str(payload.get("hard_stop_reason", "") or "")
    if is_model_output_timeout(reason) or is_model_output_timeout(hard_stop_reason):
        return True
    return reason in HARD_STOP_REASON_NAMES or hard_stop_reason in HARD_STOP_REASON_NAMES


def analyze_file(label: str, path: str) -> LogSummary:
    summary = LogSummary(label=label, path=path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            summary.raw_line_count += 1
            payload, status = parse_debug_line(line)
            if status == "ignored":
                summary.ignored_line_count += 1
                continue
            if status == "parse_error" or payload is None:
                summary.parse_error_count += 1
                continue

            summary.total_rows += 1

            tick = safe_int(payload.get("tick_count"))
            if tick is not None:
                summary.tick_values.append(tick)

            for field_name in DIST_FIELDS:
                summary.counters[field_name][category_value(payload.get(field_name))] += 1

            for field_name in NUMERIC_FIELDS:
                value = safe_float(
                    payload.get(field_name),
                    nonnegative=field_name in NONNEGATIVE_FIELDS,
                )
                if value is not None:
                    summary.values[field_name].append(value)

            angular_z = safe_float(payload.get("cmd_angular_z"))
            if angular_z is not None:
                summary.cmd_angular_valid_count += 1
                if abs(angular_z) > EPS:
                    summary.cmd_angular_nonzero_count += 1
                if angular_z > EPS:
                    summary.cmd_angular_positive_count += 1
                elif angular_z < -EPS:
                    summary.cmd_angular_negative_count += 1

            reason = str(payload.get("reason", "") or "")
            hard_stop_reason = str(payload.get("hard_stop_reason", "") or "")
            if is_hard_stop_payload(payload):
                summary.hard_stop_count += 1
            if reason == "missing_image_or_timeout" or hard_stop_reason == "missing_image_or_timeout":
                summary.missing_image_or_timeout_count += 1
            if is_model_output_timeout(reason) or is_model_output_timeout(hard_stop_reason):
                summary.model_output_timeout_count += 1

    return summary


def percentile(values: Sequence[float], percent: float) -> Optional[float]:
    """线性插值分位数；percent 取 0..100。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * (percent / 100.0)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def min_mean_max(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "min": min(values) if values else None,
        "mean": mean(values),
        "max": max(values) if values else None,
    }


def full_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "min": min(values) if values else None,
        "p50": percentile(values, 50),
        "mean": mean(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def latency_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def fmt_number(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if abs(value) < EPS:
        value = 0.0
    text = "%.6g" % value
    return text


def fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return "%.2f%%" % (value * 100.0)


def ordered_counter_items(counter: Counter) -> List[Tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def format_distribution(counter: Counter, total: int) -> str:
    if not counter:
        return "-"
    parts = []
    for key, count in ordered_counter_items(counter):
        ratio = (count / float(total)) if total else 0.0
        parts.append("%s=%d (%s)" % (key, count, fmt_ratio(ratio)))
    return ", ".join(parts)


def counter_json(counter: Counter) -> str:
    ordered = {key: count for key, count in ordered_counter_items(counter)}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def print_stat_line(name: str, stats: Dict[str, Optional[float]], keys: Iterable[str]) -> None:
    items = ["%s=%s" % (key, fmt_number(stats.get(key))) for key in keys]
    print("  %s: %s" % (name, ", ".join(items)))


def print_summary(summary: LogSummary) -> None:
    print("")
    print("=== %s: %s ===" % (summary.label, summary.path))
    print("总行数: %d" % summary.total_rows)
    if summary.parse_error_count:
        print(
            "解析提示: raw_lines=%d, ignored=%d, parse_errors=%d"
            % (
                summary.raw_line_count,
                summary.ignored_line_count,
                summary.parse_error_count,
            )
        )

    if summary.tick_values:
        print("tick_count 范围: %d .. %d" % (min(summary.tick_values), max(summary.tick_values)))
    else:
        print("tick_count 范围: -")

    for field_name in DIST_FIELDS:
        print(
            "%s 分布: %s"
            % (
                field_name,
                format_distribution(summary.counters[field_name], summary.total_rows),
            )
        )

    print_stat_line(
        "cmd_linear_x",
        min_mean_max(summary.values["cmd_linear_x"]),
        ("min", "mean", "max"),
    )

    angular_stats = min_mean_max(summary.values["cmd_angular_z"])
    nonzero_ratio = None
    if summary.cmd_angular_valid_count:
        nonzero_ratio = summary.cmd_angular_nonzero_count / float(summary.cmd_angular_valid_count)
    print(
        "  cmd_angular_z: min=%s, mean=%s, max=%s, 非零=%d (%s), 正=%d, 负=%d"
        % (
            fmt_number(angular_stats["min"]),
            fmt_number(angular_stats["mean"]),
            fmt_number(angular_stats["max"]),
            summary.cmd_angular_nonzero_count,
            fmt_ratio(nonzero_ratio),
            summary.cmd_angular_positive_count,
            summary.cmd_angular_negative_count,
        )
    )

    print_stat_line(
        "straight_keep_raw_omega",
        full_stats(summary.values["straight_keep_raw_omega"]),
        ("min", "p50", "mean", "p90", "p95", "max"),
    )
    print_stat_line(
        "pulse_recenter_error",
        full_stats(summary.values["pulse_recenter_error"]),
        ("min", "p50", "mean", "p90", "p95", "max"),
    )
    print_stat_line(
        "image_age_ms",
        latency_stats(summary.values["image_age_ms"]),
        ("p50", "p90", "p95", "max"),
    )
    print_stat_line(
        "straight_keep_age_ms",
        latency_stats(summary.values["straight_keep_age_ms"]),
        ("p50", "p90", "p95", "max"),
    )
    print_stat_line(
        "straight_keep_latency_ms",
        latency_stats(summary.values["straight_keep_latency_ms"]),
        ("p50", "p90", "p95", "max"),
    )

    print(
        "安全异常: HARD_STOP=%d, missing_image_or_timeout=%d, model_output_timeout=%d"
        % (
            summary.hard_stop_count,
            summary.missing_image_or_timeout_count,
            summary.model_output_timeout_count,
        )
    )


def build_calibration(summaries: Sequence[LogSummary]) -> Optional[Calibration]:
    by_label = {summary.label.lower(): summary for summary in summaries}
    if not all(name in by_label for name in ("left", "center", "right")):
        return None

    left_raw = percentile(by_label["left"].values["straight_keep_raw_omega"], 50)
    center_raw = percentile(by_label["center"].values["straight_keep_raw_omega"], 50)
    right_raw = percentile(by_label["right"].values["straight_keep_raw_omega"], 50)

    order_ok: Optional[bool]
    if left_raw is None or center_raw is None or right_raw is None:
        order_ok = None
    else:
        order_ok = left_raw < center_raw < right_raw

    zero_raw = center_raw
    straight_keep_bias = -center_raw if center_raw is not None else None
    return {
        "left_raw_median": left_raw,
        "center_raw_median": center_raw,
        "right_raw_median": right_raw,
        "left_center_right_ok": order_ok,
        "recommended_zero_raw": zero_raw,
        "recommended_straight_keep_bias": straight_keep_bias,
    }


def print_calibration(calibration: Optional[Calibration]) -> None:
    if calibration is None:
        return

    order_ok = calibration["left_center_right_ok"]
    if order_ok is None:
        order_text = "-"
    else:
        order_text = "true" if order_ok else "false"

    print("")
    print("=== center/left/right raw 校准建议 ===")
    print("left_raw_median: %s" % fmt_number(calibration["left_raw_median"]))
    print("center_raw_median: %s" % fmt_number(calibration["center_raw_median"]))
    print("right_raw_median: %s" % fmt_number(calibration["right_raw_median"]))
    print("是否满足 left < center < right: %s" % order_text)
    print("推荐 zero_raw: %s" % fmt_number(calibration["recommended_zero_raw"]))
    print(
        "推荐 straight_keep_bias: %s"
        % fmt_number(calibration["recommended_straight_keep_bias"])
    )


def bool_csv(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def value_csv(value: Optional[float]) -> str:
    if value is None:
        return ""
    return fmt_number(value)


def add_stats_to_row(
    row: Dict[str, Any],
    prefix: str,
    stats: Dict[str, Optional[float]],
    keys: Iterable[str],
) -> None:
    for key in keys:
        row["%s_%s" % (prefix, key)] = value_csv(stats.get(key))


def summary_to_csv_row(
    summary: LogSummary,
    calibration: Optional[Calibration],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "label": summary.label,
        "path": summary.path,
        "total_rows": summary.total_rows,
        "raw_lines": summary.raw_line_count,
        "ignored_lines": summary.ignored_line_count,
        "parse_error_count": summary.parse_error_count,
        "tick_count_min": min(summary.tick_values) if summary.tick_values else "",
        "tick_count_max": max(summary.tick_values) if summary.tick_values else "",
    }

    for field_name in DIST_FIELDS:
        row["%s_dist" % field_name] = counter_json(summary.counters[field_name])

    add_stats_to_row(
        row,
        "cmd_linear_x",
        min_mean_max(summary.values["cmd_linear_x"]),
        ("min", "mean", "max"),
    )
    add_stats_to_row(
        row,
        "cmd_angular_z",
        min_mean_max(summary.values["cmd_angular_z"]),
        ("min", "mean", "max"),
    )

    angular_ratio = None
    if summary.cmd_angular_valid_count:
        angular_ratio = summary.cmd_angular_nonzero_count / float(summary.cmd_angular_valid_count)
    row["cmd_angular_z_nonzero_count"] = summary.cmd_angular_nonzero_count
    row["cmd_angular_z_nonzero_ratio"] = value_csv(angular_ratio)
    row["cmd_angular_z_positive_count"] = summary.cmd_angular_positive_count
    row["cmd_angular_z_negative_count"] = summary.cmd_angular_negative_count

    add_stats_to_row(
        row,
        "straight_keep_raw_omega",
        full_stats(summary.values["straight_keep_raw_omega"]),
        ("min", "p50", "mean", "p90", "p95", "max"),
    )
    add_stats_to_row(
        row,
        "pulse_recenter_error",
        full_stats(summary.values["pulse_recenter_error"]),
        ("min", "p50", "mean", "p90", "p95", "max"),
    )
    add_stats_to_row(
        row,
        "image_age_ms",
        latency_stats(summary.values["image_age_ms"]),
        ("p50", "p90", "p95", "max"),
    )
    add_stats_to_row(
        row,
        "straight_keep_age_ms",
        latency_stats(summary.values["straight_keep_age_ms"]),
        ("p50", "p90", "p95", "max"),
    )
    add_stats_to_row(
        row,
        "straight_keep_latency_ms",
        latency_stats(summary.values["straight_keep_latency_ms"]),
        ("p50", "p90", "p95", "max"),
    )

    row["hard_stop_count"] = summary.hard_stop_count
    row["missing_image_or_timeout_count"] = summary.missing_image_or_timeout_count
    row["model_output_timeout_count"] = summary.model_output_timeout_count

    if calibration is not None:
        row["left_raw_median"] = value_csv(calibration["left_raw_median"])
        row["center_raw_median"] = value_csv(calibration["center_raw_median"])
        row["right_raw_median"] = value_csv(calibration["right_raw_median"])
        row["left_center_right_ok"] = bool_csv(calibration["left_center_right_ok"])
        row["recommended_zero_raw"] = value_csv(calibration["recommended_zero_raw"])
        row["recommended_straight_keep_bias"] = value_csv(
            calibration["recommended_straight_keep_bias"]
        )
    else:
        row["left_raw_median"] = ""
        row["center_raw_median"] = ""
        row["right_raw_median"] = ""
        row["left_center_right_ok"] = ""
        row["recommended_zero_raw"] = ""
        row["recommended_straight_keep_bias"] = ""

    return row


CSV_FIELDS = [
    "label",
    "path",
    "total_rows",
    "raw_lines",
    "ignored_lines",
    "parse_error_count",
    "tick_count_min",
    "tick_count_max",
    "state_dist",
    "reason_dist",
    "safety_level_dist",
    "omega_source_dist",
    "pulse_recenter_state_dist",
    "pulse_recenter_reason_dist",
    "cmd_linear_x_min",
    "cmd_linear_x_mean",
    "cmd_linear_x_max",
    "cmd_angular_z_min",
    "cmd_angular_z_mean",
    "cmd_angular_z_max",
    "cmd_angular_z_nonzero_count",
    "cmd_angular_z_nonzero_ratio",
    "cmd_angular_z_positive_count",
    "cmd_angular_z_negative_count",
    "straight_keep_raw_omega_min",
    "straight_keep_raw_omega_p50",
    "straight_keep_raw_omega_mean",
    "straight_keep_raw_omega_p90",
    "straight_keep_raw_omega_p95",
    "straight_keep_raw_omega_max",
    "pulse_recenter_error_min",
    "pulse_recenter_error_p50",
    "pulse_recenter_error_mean",
    "pulse_recenter_error_p90",
    "pulse_recenter_error_p95",
    "pulse_recenter_error_max",
    "image_age_ms_p50",
    "image_age_ms_p90",
    "image_age_ms_p95",
    "image_age_ms_max",
    "straight_keep_age_ms_p50",
    "straight_keep_age_ms_p90",
    "straight_keep_age_ms_p95",
    "straight_keep_age_ms_max",
    "straight_keep_latency_ms_p50",
    "straight_keep_latency_ms_p90",
    "straight_keep_latency_ms_p95",
    "straight_keep_latency_ms_max",
    "hard_stop_count",
    "missing_image_or_timeout_count",
    "model_output_timeout_count",
    "left_raw_median",
    "center_raw_median",
    "right_raw_median",
    "left_center_right_ok",
    "recommended_zero_raw",
    "recommended_straight_keep_bias",
]


def write_csv(
    out_csv: str,
    summaries: Sequence[LogSummary],
    calibration: Optional[Calibration],
) -> None:
    out_path = Path(out_csv)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        os.makedirs(str(out_path.parent), exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary_to_csv_row(summary, calibration))


def main() -> int:
    args = parse_args()
    try:
        specs = parse_log_specs(args.logs)
    except ValueError as exc:
        raise SystemExit("参数错误: %s" % exc)

    summaries = []
    for label, path in specs:
        try:
            summaries.append(analyze_file(label, path))
        except OSError as exc:
            raise SystemExit("读取日志失败: %s=%s (%s)" % (label, path, exc))
    calibration = build_calibration(summaries)

    for summary in summaries:
        print_summary(summary)
    print_calibration(calibration)

    if args.out_csv:
        write_csv(args.out_csv, summaries, calibration)
        print("")
        print("CSV 已保存: %s" % args.out_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
