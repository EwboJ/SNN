#!/usr/bin/env python3
"""Compact /nav/debug watcher.

Usage:
  ros2 topic echo /nav/debug --full-length --field data \
      | python scripts/watch_nav_debug.py --mode summary
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List


FIELDS_BY_MODE: Dict[str, List[str]] = {
    "summary": [
        "state",
        "cmd_linear_x",
        "cmd_angular_z",
        "omega_source",
        "trigger_pred",
        "stage3_pred",
        "image_age_ms",
        "straight_keep_age_ms",
        "reason",
    ],
    "omega": [
        "state",
        "straight_keep_raw_omega",
        "straight_keep_bias",
        "straight_keep_scale",
        "straight_keep_after_bias",
        "straight_keep_after_scale",
        "straight_keep_after_deadband",
        "straight_keep_after_clip",
        "straight_keep_final_omega",
        "cmd_angular_z",
        "omega_source",
    ],
    "stale": [
        "state",
        "omega_source",
        "cmd_from_new_inference",
        "cmd_from_cached_output",
        "stale_omega_suppressed",
        "stale_omega_before",
        "stale_omega_after",
        "straight_keep_age_ms",
        "latest_inference_age_ms",
        "max_omega_hold_sec",
    ],
    # slow_safe_stage1c: 观察脉冲式纠偏状态机与最终发布速度。
    "pulse": [
        "state",
        "cmd_linear_x",
        "cmd_angular_z",
        "omega_source",
        "straight_keep_raw_omega",
        "straight_keep_bias",
        "pulse_recenter_enable",
        "pulse_recenter_state",
        "pulse_recenter_error",
        "pulse_recenter_dir",
        "pulse_recenter_step_count",
        "pulse_recenter_enter_count",
        "pulse_recenter_exit_count",
        "pulse_recenter_cooldown_count",
        "pulse_recenter_reason",
        "straight_keep_age_ms",
        "image_age_ms",
        "reason",
    ],
    "timing": [
        "image_age_ms",
        "straight_keep_age_ms",
        "stage3_age_ms",
        "trigger_age_ms",
        "junction_age_ms",
        "straight_keep_busy",
        "stage3_busy",
        "trigger_busy",
        "junction_busy",
        "tick_count",
        "image_rx_count",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read JSON lines from ros2 topic echo /nav/debug --field data."
    )
    parser.add_argument(
        "--mode",
        choices=sorted(FIELDS_BY_MODE.keys()),
        default="summary",
        help="Field group to print.",
    )
    return parser.parse_args()


def parse_debug_line(line: str) -> Dict[str, Any] | None:
    text = line.strip()
    if not text or text == "---":
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict):
        return None
    return payload


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value)
    if not text:
        return "-"
    if any(ch.isspace() for ch in text):
        return json.dumps(text, ensure_ascii=False)
    return text


def format_row(payload: Dict[str, Any], fields: Iterable[str]) -> str:
    return " ".join(f"{field}={format_value(payload.get(field))}" for field in fields)


def main() -> int:
    args = parse_args()
    fields = FIELDS_BY_MODE[args.mode]

    for line in sys.stdin:
        payload = parse_debug_line(line)
        if payload is None:
            continue
        print(format_row(payload, fields), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
