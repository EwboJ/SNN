#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线测试 corridor SNN 推理延迟。

使用示例:
  python3 scripts/benchmark_corridor_infer.py \
    --module straight_keep \
    --ckpt checkpoint/.../best_model.ckpt \
    --image demo.jpg \
    --device cuda:0 \
    --warmup 20 \
    --iters 100

连续测试多个模型:
  python3 scripts/benchmark_corridor_infer.py \
    --modules straight_keep,approach_trigger,stage3,junction_lr \
    --ckpts path1,path2,path3,path4 \
    --image demo.jpg \
    --device cuda:0 \
    --warmup 20 \
    --iters 100 \
    --out_csv benchmark.csv

说明:
  - 不依赖 ROS2，直接复用 inference/corridor_module_infer.py 中的推理类。
  - 若推理类支持 set_profiling(True)，会读取 predict() 返回的 timing 字段。
  - 若没有 timing 字段，则至少统计 predict() 总耗时 total_ms。
  - CUDA 下每轮计时前后都会 torch.cuda.synchronize()，用于获得真实同步耗时。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.corridor_module_infer import (  # noqa: E402
    ApproachTriggerInfer,
    JunctionLRInfer,
    Stage3Infer,
    StraightKeepInfer,
)


SUPPORTED_MODULES = ("straight_keep", "approach_trigger", "stage3", "junction_lr")
TIMING_FIELDS = (
    "total_ms",
    "preprocess_ms",
    "to_device_ms",
    "forward_ms",
    "output_to_cpu_ms",
)
STAT_FIELDS = ("min", "mean", "p50", "p90", "p95", "max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark corridor_module_infer.py 模型推理延迟，不依赖 ROS2。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    one_or_many = parser.add_argument_group("单模型或多模型输入")
    one_or_many.add_argument(
        "--module",
        choices=SUPPORTED_MODULES,
        help="单模型名称。",
    )
    one_or_many.add_argument(
        "--ckpt",
        help="单模型 checkpoint 路径。",
    )
    one_or_many.add_argument(
        "--modules",
        help="逗号分隔的模型名称，例如 straight_keep,approach_trigger,stage3,junction_lr。",
    )
    one_or_many.add_argument(
        "--ckpts",
        help="逗号分隔的 checkpoint 路径，数量需与 --modules 一致。",
    )
    parser.add_argument("--image", required=True, help="输入图片路径。")
    parser.add_argument(
        "--device",
        default=None,
        help="推理设备，例如 cpu / cuda:0；默认沿用推理类自动选择。",
    )
    parser.add_argument("--warmup", type=int, default=20, help="预热轮数。")
    parser.add_argument("--iters", type=int, default=100, help="正式计时轮数。")
    parser.add_argument("--out_csv", default=None, help="可选 summary CSV 输出路径。")
    return parser.parse_args()


def split_csv_arg(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_benchmark_specs(args: argparse.Namespace) -> List[Tuple[str, str]]:
    modules = split_csv_arg(args.modules)
    ckpts = split_csv_arg(args.ckpts)
    if modules or ckpts:
        if not modules:
            raise SystemExit("参数错误: 使用 --ckpts 时必须同时提供 --modules")
        if not ckpts:
            raise SystemExit("参数错误: 使用 --modules 时必须同时提供 --ckpts")
        if len(modules) != len(ckpts):
            raise SystemExit(
                "参数错误: --modules 数量(%d)与 --ckpts 数量(%d)不一致"
                % (len(modules), len(ckpts))
            )
    else:
        if not args.module or not args.ckpt:
            raise SystemExit("参数错误: 请提供 --module/--ckpt 或 --modules/--ckpts")
        modules = [args.module]
        ckpts = [args.ckpt]

    specs: List[Tuple[str, str]] = []
    for module_name, ckpt_path in zip(modules, ckpts):
        module_name = module_name.strip()
        if module_name not in SUPPORTED_MODULES:
            raise SystemExit(
                "参数错误: 不支持的 module=%s，可选: %s"
                % (module_name, ",".join(SUPPORTED_MODULES))
            )
        if not ckpt_path:
            raise SystemExit("参数错误: %s 的 ckpt 路径为空" % module_name)
        specs.append((module_name, ckpt_path))
    return specs


def build_infer(module_name: str, ckpt_path: str, device: Optional[str]) -> Any:
    if module_name == "straight_keep":
        return StraightKeepInfer(ckpt_path=ckpt_path, device=device)
    if module_name == "approach_trigger":
        return ApproachTriggerInfer(ckpt_path=ckpt_path, device=device)
    if module_name == "stage3":
        return Stage3Infer(ckpt_path=ckpt_path, device=device)
    if module_name == "junction_lr":
        return JunctionLRInfer(ckpt_path=ckpt_path, device=device)
    raise ValueError("不支持的 module: %s" % module_name)


def resolve_device_for_sync(device_arg: Optional[str], infer: Any) -> Optional[torch.device]:
    if device_arg:
        device = torch.device(device_arg)
    else:
        device = getattr(infer, "device", None)
        if device is None:
            return None
        device = torch.device(device)
    return device


def sync_cuda(device: Optional[torch.device]) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def extract_timing(result: Any, fallback_total_ms: float) -> Dict[str, float]:
    timing: Dict[str, float] = {}
    if isinstance(result, dict) and isinstance(result.get("timing"), dict):
        for key, value in result["timing"].items():
            number = safe_float(value)
            if isinstance(key, str) and number is not None:
                timing[key] = number
    if "total_ms" not in timing:
        timing["total_ms"] = float(fallback_total_ms)
    return timing


def percentile(values: Sequence[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (percent / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {key: None for key in STAT_FIELDS}
    return {
        "min": min(values),
        "mean": sum(values) / float(len(values)),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return "%.3f" % value


def benchmark_one(
    module_name: str,
    ckpt_path: str,
    image: Image.Image,
    device_arg: Optional[str],
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    infer = build_infer(module_name, ckpt_path, device_arg)
    if hasattr(infer, "set_profiling"):
        infer.set_profiling(True)
    device = resolve_device_for_sync(device_arg, infer)

    for _ in range(max(0, int(warmup))):
        sync_cuda(device)
        infer.predict(image)
        sync_cuda(device)

    samples: Dict[str, List[float]] = {field: [] for field in TIMING_FIELDS}
    for _ in range(max(1, int(iters))):
        sync_cuda(device)
        start = time.perf_counter()
        result = infer.predict(image)
        sync_cuda(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        timing = extract_timing(result, fallback_total_ms=elapsed_ms)
        for field in TIMING_FIELDS:
            value = timing.get(field)
            if value is not None:
                samples[field].append(float(value))

    summary = {field: summarize(samples[field]) for field in TIMING_FIELDS}
    return {
        "module": module_name,
        "ckpt": ckpt_path,
        "device": str(device) if device is not None else "",
        "warmup": int(warmup),
        "iters": int(iters),
        "summary": summary,
    }


def print_summary(result: Dict[str, Any]) -> None:
    print("")
    print("=== %s ===" % result["module"])
    print("ckpt: %s" % result["ckpt"])
    print("device: %s, warmup=%d, iters=%d" % (result["device"], result["warmup"], result["iters"]))
    for field in TIMING_FIELDS:
        stats = result["summary"][field]
        print(
            "%s: min=%s mean=%s p50=%s p90=%s p95=%s max=%s"
            % (
                field,
                fmt_ms(stats["min"]),
                fmt_ms(stats["mean"]),
                fmt_ms(stats["p50"]),
                fmt_ms(stats["p90"]),
                fmt_ms(stats["p95"]),
                fmt_ms(stats["max"]),
            )
        )


def write_csv(path: str, results: Sequence[Dict[str, Any]]) -> None:
    out_path = Path(path)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        os.makedirs(str(out_path.parent), exist_ok=True)

    fieldnames = ["module", "ckpt", "device", "warmup", "iters"]
    for timing_field in TIMING_FIELDS:
        for stat_field in STAT_FIELDS:
            fieldnames.append("%s_%s" % (timing_field, stat_field))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: Dict[str, Any] = {
                "module": result["module"],
                "ckpt": result["ckpt"],
                "device": result["device"],
                "warmup": result["warmup"],
                "iters": result["iters"],
            }
            for timing_field in TIMING_FIELDS:
                stats = result["summary"][timing_field]
                for stat_field in STAT_FIELDS:
                    value = stats[stat_field]
                    row["%s_%s" % (timing_field, stat_field)] = (
                        "" if value is None else "%.6f" % value
                    )
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    specs = parse_benchmark_specs(args)
    image = Image.open(args.image).convert("RGB")

    results = []
    for module_name, ckpt_path in specs:
        result = benchmark_one(
            module_name=module_name,
            ckpt_path=ckpt_path,
            image=image,
            device_arg=args.device,
            warmup=max(0, int(args.warmup)),
            iters=max(1, int(args.iters)),
        )
        results.append(result)
        print_summary(result)

    if args.out_csv:
        write_csv(args.out_csv, results)
        print("")
        print("CSV 已保存: %s" % args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
