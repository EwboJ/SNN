"""
approach_trigger 二分类数据集快速核查脚本
=======================================
用于快速检查类似 approach_trigger_v1_sameenv_shortwin 的派生结果是否合理：
1) 统计样本数量（全局 / split / run）
2) 输出每个 run 的样本数 TopN
3) 随机抽样 run 生成预览图（strip + 标签彩条 + t_rel_ms）

目录结构（输入）:
  data_root/{train,val,test}/{run_name}/
    - images/
    - labels.csv
    - meta.json
"""

import os
import csv
import json
import random
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
from PIL import Image, ImageDraw

# matplotlib 延迟导入：避免仅做统计时因绘图库环境问题直接崩溃
plt = None
mpatches = None
gridspec = None


SPLITS = ["train", "val", "test"]

# 颜色方案：NearTurnEvent 橙红，Straight 蓝绿
LABEL_COLORS = {
    "Straight": "#2E86DE",       # 蓝
    "NearTurnEvent": "#E67E22",  # 橙
}

BORDER_COLORS = {
    "Straight": "#1B4F72",       # 深蓝
    "NearTurnEvent": "#C0392B",  # 红
}


def ensure_matplotlib():
    """
    延迟初始化 matplotlib；成功返回 True，失败返回 False。
    """
    global plt, mpatches, gridspec
    if plt is not None and mpatches is not None and gridspec is not None:
        return True
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import matplotlib.patches as _mpatches
        import matplotlib.gridspec as _gridspec
    except Exception as e:
        print(f"[Warn] matplotlib unavailable, preview generation disabled: {e}")
        return False

    # 中文 / 英文字体兼容
    for font in ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]:
        try:
            _plt.rcParams["font.sans-serif"] = [font] + _plt.rcParams["font.sans-serif"]
            break
        except Exception:
            continue
    _plt.rcParams["axes.unicode_minus"] = False
    _plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130})

    plt = _plt
    mpatches = _mpatches
    gridspec = _gridspec
    return True


def safe_int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_splits(split_arg):
    """解析 --split 参数；支持单个或逗号分隔。"""
    if not split_arg:
        return list(SPLITS)
    raw = [s.strip() for s in str(split_arg).split(",")]
    selected = []
    for s in raw:
        if s in SPLITS and s not in selected:
            selected.append(s)
    return selected


def load_labels_csv(csv_path):
    """读取单个 run 的 labels.csv。"""
    rows = []
    if not os.path.isfile(csv_path):
        return rows

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for ridx, row in enumerate(reader):
            rows.append({
                "row_idx": ridx,
                "image_name": (row.get("image_name") or "").strip(),
                "label_id": safe_int(row.get("label_id"), None),
                "label_name": (row.get("label_name") or "").strip(),
                "orig_action_name": (row.get("orig_action_name") or "").strip(),
                "timestamp_ns": safe_int(row.get("timestamp_ns"), None),
                "t_rel_ms_raw": row.get("t_rel_ms", ""),
                "t_rel_ms": safe_float(row.get("t_rel_ms", ""), None),
                "valid": safe_int(row.get("valid"), None),
                "source_run": (row.get("source_run") or "").strip(),
                "source_split": (row.get("source_split") or "").strip(),
            })

    # 按时间排序（时间戳缺失时保持原行序）
    rows.sort(
        key=lambda r: (
            r["timestamp_ns"] is None,
            r["timestamp_ns"] if r["timestamp_ns"] is not None else r["row_idx"],
            r["row_idx"],
        )
    )
    return rows


def scan_runs(data_root, selected_splits):
    """扫描数据集并读取每个 run 的 labels。"""
    runs = []
    warnings = []

    for split in selected_splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            warnings.append(f"[Warn] split dir not found: {split_dir}")
            continue

        for run_name in sorted(os.listdir(split_dir)):
            run_dir = os.path.join(split_dir, run_name)
            if not os.path.isdir(run_dir):
                continue

            csv_path = os.path.join(run_dir, "labels.csv")
            img_dir = os.path.join(run_dir, "images")
            meta_path = os.path.join(run_dir, "meta.json")
            if not os.path.isfile(csv_path):
                warnings.append(f"[Warn] missing labels.csv, skip run: {split}/{run_name}")
                continue
            if not os.path.isdir(img_dir):
                warnings.append(f"[Warn] missing images/, skip run: {split}/{run_name}")
                continue
            if not os.path.isfile(meta_path):
                # meta.json 不是强依赖，但给出提示
                warnings.append(f"[Warn] missing meta.json: {split}/{run_name}")

            frames = load_labels_csv(csv_path)
            if len(frames) == 0:
                warnings.append(f"[Warn] empty labels.csv, skip run: {split}/{run_name}")
                continue

            runs.append({
                "split": split,
                "run_name": run_name,
                "run_dir": run_dir,
                "img_dir": img_dir,
                "frames": frames,
            })

    return runs, warnings


def build_statistics(runs):
    """统计全局 / split / run 级别数量分布。"""
    stats = OrderedDict()
    stats["global"] = {
        "total_samples": 0,
        "label_distribution": defaultdict(int),
    }
    stats["splits"] = OrderedDict()
    stats["runs"] = OrderedDict()

    for sp in SPLITS:
        stats["splits"][sp] = {
            "run_count": 0,
            "sample_count": 0,
            "label_distribution": defaultdict(int),
        }

    for run in runs:
        split = run["split"]
        run_name = run["run_name"]
        frames = run["frames"]

        run_dist = defaultdict(int)
        for fr in frames:
            label_name = fr["label_name"] if fr["label_name"] else f"class_{fr['label_id']}"
            run_dist[label_name] += 1
            stats["global"]["label_distribution"][label_name] += 1
            stats["splits"][split]["label_distribution"][label_name] += 1

        run_count = len(frames)
        stats["global"]["total_samples"] += run_count
        stats["splits"][split]["run_count"] += 1
        stats["splits"][split]["sample_count"] += run_count
        stats["runs"][f"{split}/{run_name}"] = {
            "split": split,
            "run_name": run_name,
            "sample_count": run_count,
            "label_distribution": dict(run_dist),
        }

    # 转成普通 dict，便于 JSON 序列化
    stats["global"]["label_distribution"] = dict(stats["global"]["label_distribution"])
    for sp in SPLITS:
        stats["splits"][sp]["label_distribution"] = dict(stats["splits"][sp]["label_distribution"])
    return stats


def print_statistics(stats, topn=12):
    """终端打印统计摘要。"""
    print("=" * 88)
    print(" Dataset Inspection Summary")
    print("=" * 88)
    print(f"Global total samples: {stats['global']['total_samples']}")
    print(
        "Global label dist   : "
        + ", ".join(
            f"{k}={v}" for k, v in sorted(stats["global"]["label_distribution"].items())
        )
    )
    print("-" * 88)

    for sp in SPLITS:
        sp_info = stats["splits"][sp]
        print(
            f"[{sp}] runs={sp_info['run_count']}, samples={sp_info['sample_count']}, "
            f"labels={sp_info['label_distribution']}"
        )

    run_items = list(stats["runs"].items())
    run_items.sort(key=lambda kv: kv[1]["sample_count"], reverse=True)
    show_n = min(topn, len(run_items))
    print("-" * 88)
    print(f"Top {show_n} runs by sample_count:")
    for i in range(show_n):
        k, v = run_items[i]
        print(f"  {i+1:02d}. {k:40s} total={v['sample_count']}")
    print("=" * 88)


def _make_missing_thumbnail(thumb_w, thumb_h, text="MISSING"):
    """构造缺图占位图，避免中断流程。"""
    img = Image.new("RGB", (thumb_w, thumb_h), color=(35, 35, 35))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, thumb_w - 1, thumb_h - 1], outline=(220, 60, 60), width=2)
    draw.line([0, 0, thumb_w - 1, thumb_h - 1], fill=(220, 60, 60), width=2)
    draw.line([thumb_w - 1, 0, 0, thumb_h - 1], fill=(220, 60, 60), width=2)
    draw.text((5, max(2, thumb_h // 2 - 8)), text, fill=(230, 230, 230))
    return np.array(img)


def sample_frames_for_preview(frames, frames_per_run):
    """按时间顺序均匀抽样若干帧用于预览。"""
    n = len(frames)
    if n <= 0:
        return []
    if n <= frames_per_run:
        idxs = list(range(n))
    else:
        idxs = np.linspace(0, n - 1, frames_per_run, dtype=int).tolist()
    return [frames[i] for i in idxs]


def build_strip_images(run_info, frames_for_preview, thumb_w=64, thumb_h=64):
    """加载缩略图并拼接为 strip。"""
    tiles = []
    missing_count = 0

    for fr in frames_for_preview:
        img_path = os.path.join(run_info["img_dir"], fr["image_name"])
        if os.path.isfile(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
                tiles.append(np.array(img))
                continue
            except Exception:
                pass
        tiles.append(_make_missing_thumbnail(thumb_w, thumb_h))
        missing_count += 1

    if not tiles:
        return None, 0
    strip = np.concatenate(tiles, axis=1)
    return strip, missing_count


def plot_run_preview(run_info, frames_per_run, out_path):
    """绘制单个 run 预览图（上：图像 strip，下：标签彩条 + t_rel_ms）。"""
    if not ensure_matplotlib():
        return {
            "ok": False,
            "reason": "matplotlib_unavailable",
            "missing_images": 0,
            "shown_frames": 0,
        }

    frames = run_info["frames"]
    sampled = sample_frames_for_preview(frames, frames_per_run=frames_per_run)
    if len(sampled) == 0:
        return {
            "ok": False,
            "reason": "empty_run",
            "missing_images": 0,
            "shown_frames": 0,
        }

    strip, missing_count = build_strip_images(run_info, sampled, thumb_w=64, thumb_h=64)
    if strip is None:
        return {
            "ok": False,
            "reason": "strip_failed",
            "missing_images": missing_count,
            "shown_frames": 0,
        }

    n = len(sampled)
    fig_w = max(12, n * 0.65)
    fig_h = 4.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.2, 1.2], hspace=0.15)

    # 上半部分：图片 strip
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(strip, aspect="auto")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(
        f"{run_info['split']}/{run_info['run_name']}  "
        f"(total={len(frames)}, shown={n}, missing_img={missing_count})",
        fontsize=11,
        fontweight="bold",
    )

    # 每帧边框（NearTurnEvent 橙红，Straight 蓝绿）
    tile_w = strip.shape[1] / n
    tile_h = strip.shape[0]
    for i, fr in enumerate(sampled):
        label_name = fr["label_name"] if fr["label_name"] else "Unknown"
        edge_color = BORDER_COLORS.get(label_name, "#7F8C8D")
        rect = plt.Rectangle(
            (i * tile_w, 0),
            tile_w - 1,
            tile_h - 1,
            linewidth=2.2,
            edgecolor=edge_color,
            facecolor="none",
        )
        ax_img.add_patch(rect)

    # 下半部分：标签彩条 + t_rel_ms
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_xlim(0, n)
    ax_bar.set_ylim(-0.42, 1.02)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Frame index in preview strip", fontsize=9)

    legend_seen = OrderedDict()
    for i, fr in enumerate(sampled):
        label_name = fr["label_name"] if fr["label_name"] else "Unknown"
        color = LABEL_COLORS.get(label_name, "#95A5A6")
        ax_bar.barh(0.5, 1.0, left=i, height=0.75, color=color, alpha=0.88)

        if label_name not in legend_seen:
            legend_seen[label_name] = color

        # 若 t_rel_ms 存在，显示在彩条下方
        if fr["t_rel_ms"] is not None:
            txt = f"{fr['t_rel_ms']:.0f}ms"
            if n <= 24 or i % max(1, n // 12) == 0:
                ax_bar.text(i + 0.5, -0.06, txt, ha="center", va="top", fontsize=7, color="#333333")

    handles = [mpatches.Patch(color=c, label=ln) for ln, c in legend_seen.items()]
    if handles:
        ax_bar.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, ncol=max(1, len(handles)))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "ok": True,
        "reason": "",
        "missing_images": missing_count,
        "shown_frames": n,
    }


def create_previews(runs, out_dir, max_runs=8, frames_per_run=20, seed=42):
    """随机抽取 run 生成预览图。"""
    os.makedirs(out_dir, exist_ok=True)
    if not ensure_matplotlib():
        return {
            "selected_run_count": 0,
            "preview_files": [],
            "total_missing_images_in_preview": 0,
            "preview_disabled_reason": "matplotlib_unavailable",
        }
    rng = random.Random(seed)

    if len(runs) <= max_runs:
        selected = list(runs)
    else:
        selected = rng.sample(runs, k=max_runs)

    selected.sort(key=lambda r: (r["split"], r["run_name"]))

    preview_stats = {
        "selected_run_count": len(selected),
        "preview_files": [],
        "total_missing_images_in_preview": 0,
    }

    for run in selected:
        filename = f"{run['split']}__{run['run_name']}.png"
        out_path = os.path.join(out_dir, filename)
        res = plot_run_preview(run, frames_per_run=frames_per_run, out_path=out_path)
        if res["ok"]:
            preview_stats["preview_files"].append({
                "run": f"{run['split']}/{run['run_name']}",
                "file": filename,
                "shown_frames": res["shown_frames"],
                "missing_images": res["missing_images"],
            })
            preview_stats["total_missing_images_in_preview"] += res["missing_images"]
        else:
            preview_stats["preview_files"].append({
                "run": f"{run['split']}/{run['run_name']}",
                "file": "",
                "error": res["reason"],
            })
    return preview_stats


def run_inspection(args):
    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, "previews")
    os.makedirs(preview_dir, exist_ok=True)

    selected_splits = parse_splits(args.split)
    if not selected_splits:
        raise ValueError("--split has no valid split name, expected one of train,val,test")

    print("=" * 88)
    print(" Inspect approach_trigger dataset")
    print("=" * 88)
    print(f"data_root       : {os.path.abspath(args.data_root)}")
    print(f"out_dir         : {os.path.abspath(args.out_dir)}")
    print(f"splits          : {selected_splits}")
    print(f"max_runs        : {args.max_runs}")
    print(f"frames_per_run  : {args.frames_per_run}")
    print(f"seed            : {args.seed}")
    print("=" * 88)

    runs, warnings = scan_runs(args.data_root, selected_splits=selected_splits)
    for w in warnings:
        print(w)

    if len(runs) == 0:
        print("[Warn] No valid runs found. Exit.")
        return

    stats = build_statistics(runs)
    print_statistics(stats, topn=12)

    print("\n[Preview] Generating preview images ...")
    preview_stats = create_previews(
        runs=runs,
        out_dir=preview_dir,
        max_runs=args.max_runs,
        frames_per_run=args.frames_per_run,
        seed=args.seed,
    )
    print(f"  selected runs          : {preview_stats['selected_run_count']}")
    print(f"  generated previews     : {len([x for x in preview_stats['preview_files'] if x.get('file')])}")
    print(f"  missing images in prev : {preview_stats['total_missing_images_in_preview']}")
    print(f"  preview dir            : {os.path.abspath(preview_dir)}")

    summary = OrderedDict()
    summary["data_root"] = os.path.abspath(args.data_root)
    summary["out_dir"] = os.path.abspath(args.out_dir)
    summary["selected_splits"] = selected_splits
    summary["max_runs"] = args.max_runs
    summary["frames_per_run"] = args.frames_per_run
    summary["seed"] = args.seed
    summary["warnings"] = warnings
    summary["stats"] = stats
    summary["preview"] = preview_stats

    summary_path = os.path.join(args.out_dir, "inspect_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] {summary_path}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Quickly inspect approach_trigger binary dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/stage1_v3/approach_trigger_v1_sameenv_shortwin",
        help="Dataset root path.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/inspect_approach_trigger",
        help="Output directory for summary and preview images.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split filter: train/val/test or comma-separated list.",
    )
    parser.add_argument("--max_runs", type=int, default=8)
    parser.add_argument("--frames_per_run", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.max_runs <= 0:
        raise ValueError("--max_runs must be >= 1")
    if args.frames_per_run <= 0:
        raise ValueError("--frames_per_run must be >= 1")

    run_inspection(args)


if __name__ == "__main__":
    main()
