"""
Derive approach-trigger binary dataset from corridor_task runs.

Target dataset:
  approach_trigger_v1_sameenv_shortwin

Labels:
  0 -> Straight
  1 -> NearTurnEvent

Primary source:
  ./data/corridor_stage3_rawsplit

Optional auxiliary source:
  --straight_root (disabled by default)
"""

import os
import csv
import json
import shutil
import argparse
from collections import OrderedDict, defaultdict


SPLITS = ["train", "val", "test"]
DATASET_NAME = "approach_trigger_v1_sameenv_shortwin"
LABEL_ID_TO_NAME = OrderedDict([
    (0, "Straight"),
    (1, "NearTurnEvent"),
])
TURN_ACTIONS = {"left", "right"}
STRAIGHT_ACTIONS = {"follow", "forward"}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


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


def normalize_action_name(name):
    if name is None:
        return ""
    return str(name).strip().lower()


def load_labels_csv(csv_path, valid_only=True, source_split="", source_run=""):
    """
    Load labels.csv safely.

    Returns:
      frames: list[dict]
      skipped_rows: int
    """
    frames = []
    skipped_rows = 0

    if not os.path.isfile(csv_path):
        return frames, skipped_rows

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for ridx, row in enumerate(reader):
            image_name = (row.get("image_name") or "").strip()
            if not image_name:
                skipped_rows += 1
                continue

            ts_ns = safe_int(row.get("timestamp_ns"), None)
            if ts_ns is None:
                skipped_rows += 1
                continue

            valid_val = safe_int(row.get("valid", 1), 1)
            valid = 1 if valid_val == 1 else 0
            if valid_only and valid != 1:
                continue

            action_name_raw = (row.get("action_name") or "").strip()
            frame = {
                "row_idx": ridx,
                "image_name": image_name,
                "action_id": safe_int(row.get("action_id"), None),
                "action_name": action_name_raw,
                "action_name_norm": normalize_action_name(action_name_raw),
                "timestamp_ns": ts_ns,
                "linear_x": safe_float(row.get("linear_x"), None),
                "angular_z": safe_float(row.get("angular_z"), None),
                "time_diff_ms": safe_float(row.get("time_diff_ms"), None),
                "valid": valid,
                "source_split": source_split,
                "source_run": source_run,
            }
            frames.append(frame)

    return frames, skipped_rows


def detect_turns(frames, min_turn_k=3):
    """
    Detect turn events from consecutive Left/Right segments.

    Rule:
      consecutive K frames of same direction Left/Right => one turn event.
    """
    turns = []
    n = len(frames)
    if n <= 0 or min_turn_k <= 0:
        return turns

    i = 0
    turn_id = 0
    while i < n:
        action = frames[i]["action_name_norm"]
        if action not in TURN_ACTIONS:
            i += 1
            continue

        j = i
        while j + 1 < n and frames[j + 1]["action_name_norm"] == action:
            j += 1

        seg_len = j - i + 1
        if seg_len >= min_turn_k:
            t_on = frames[i]["timestamp_ns"]
            t_off = frames[j]["timestamp_ns"]
            turns.append({
                "turn_id": turn_id,
                "turn_dir": "Left" if action == "left" else "Right",
                "turn_on_ns": t_on,
                "turn_off_ns": t_off,
                "idx_on": frames[i]["row_idx"],
                "idx_off": frames[j]["row_idx"],
                "frame_idx_on": i,
                "frame_idx_off": j,
            })
            turn_id += 1

        i = j + 1

    return turns


def _dist_to_turn_ms(ts_ns, turn):
    on = turn["turn_on_ns"]
    off = turn["turn_off_ns"]
    if ts_ns < on:
        return (on - ts_ns) / 1e6
    if ts_ns > off:
        return (ts_ns - off) / 1e6
    return 0.0


def _is_turn_frame(ts_ns, turn):
    return turn["turn_on_ns"] <= ts_ns <= turn["turn_off_ns"]


def _is_post_turn_exclude(ts_ns, turn, post_turn_exclude_ms):
    return turn["turn_off_ns"] < ts_ns <= turn["turn_off_ns"] + int(post_turn_exclude_ms * 1e6)


def _find_positive_match(ts_ns, turns, pre_turn_ms, pre_turn_end_ms):
    """
    Return (matched_turn, t_rel_ms) if ts is in positive window.
    """
    matched = None
    best_abs_rel = None

    for turn in turns:
        win_start = turn["turn_on_ns"] - int(pre_turn_ms * 1e6)
        win_end = turn["turn_on_ns"] - int(pre_turn_end_ms * 1e6)
        if win_start <= ts_ns <= win_end:
            rel_ms = (ts_ns - turn["turn_on_ns"]) / 1e6
            abs_rel = abs(rel_ms)
            if best_abs_rel is None or abs_rel < best_abs_rel:
                matched = (turn, rel_ms)
                best_abs_rel = abs_rel

    return matched


def collect_positive_negative_frames(
    frames,
    turns,
    pre_turn_ms=700,
    pre_turn_end_ms=100,
    safe_margin_ms=2500,
    post_turn_exclude_ms=1000,
    stride=3,
):
    """
    Collect explicit positive/negative samples and ignore ambiguous frames.
    """
    if stride <= 0:
        stride = 1

    positive_samples = []
    positive_frame_ids = set()

    for idx, frame in enumerate(frames):
        ts_ns = frame["timestamp_ns"]
        pm = _find_positive_match(ts_ns, turns, pre_turn_ms, pre_turn_end_ms)
        if pm is None:
            continue
        turn, rel_ms = pm
        positive_frame_ids.add(idx)
        positive_samples.append({
            "frame_idx": idx,
            "frame": frame,
            "label_id": 1,
            "label_name": LABEL_ID_TO_NAME[1],
            "t_rel_ms": rel_ms,
            "matched_turn_id": turn["turn_id"],
        })

    negative_candidates = []
    for idx, frame in enumerate(frames):
        if idx in positive_frame_ids:
            continue

        ts_ns = frame["timestamp_ns"]

        in_turn = any(_is_turn_frame(ts_ns, t) for t in turns)
        if in_turn:
            continue

        in_post_exclude = any(
            _is_post_turn_exclude(ts_ns, t, post_turn_exclude_ms) for t in turns
        )
        if in_post_exclude:
            continue

        if frame["action_name_norm"] not in STRAIGHT_ACTIONS:
            continue

        min_dist_ms = min(_dist_to_turn_ms(ts_ns, t) for t in turns) if turns else float("inf")
        if min_dist_ms < float(safe_margin_ms):
            continue

        negative_candidates.append({
            "frame_idx": idx,
            "frame": frame,
            "label_id": 0,
            "label_name": LABEL_ID_TO_NAME[0],
            "t_rel_ms": "",
            "matched_turn_id": None,
        })

    negative_samples = negative_candidates[::stride]
    samples = positive_samples + negative_samples
    samples.sort(key=lambda s: s["frame"]["timestamp_ns"])

    stats = {
        "positive_count": len(positive_samples),
        "negative_candidate_count": len(negative_candidates),
        "negative_count_after_stride": len(negative_samples),
        "selected_total": len(samples),
    }
    return samples, stats


def _format_t_rel_ms(v):
    if v == "" or v is None:
        return ""
    return f"{float(v):.3f}".rstrip("0").rstrip(".")


def _safe_link_or_copy(src_path, dst_path, copy_mode="symlink"):
    """
    copy_mode:
      - copy
      - symlink (fallback to copy on failure)
    """
    if copy_mode == "copy":
        shutil.copy2(src_path, dst_path)
        return "copy"

    try:
        os.symlink(os.path.abspath(src_path), dst_path)
        return "symlink"
    except Exception:
        shutil.copy2(src_path, dst_path)
        return "copy_fallback"


def _ensure_run_output_dir(dst_run_dir, force=False):
    if os.path.exists(dst_run_dir):
        if not force:
            return False
        shutil.rmtree(dst_run_dir)
    os.makedirs(os.path.join(dst_run_dir, "images"), exist_ok=True)
    return True


def write_derived_run(
    src_run_dir,
    dst_run_dir,
    split,
    run_name,
    samples,
    turns,
    derive_config,
    source_type="sameenv",
    force=False,
    copy_mode="symlink",
    extra_meta=None,
):
    """
    Write one derived run:
      - images/
      - labels.csv
      - meta.json
    """
    if extra_meta is None:
        extra_meta = {}

    created = _ensure_run_output_dir(dst_run_dir, force=force)
    if not created:
        print(f"  [Skip] {split}/{run_name}: dst exists (use --force to overwrite)")
        return None

    src_img_dir = os.path.join(src_run_dir, "images")
    dst_img_dir = os.path.join(dst_run_dir, "images")
    if not os.path.isdir(src_img_dir):
        print(f"  [Warn] {split}/{run_name}: missing images/ in source, skip run")
        return None

    fieldnames = [
        "image_name",
        "label_id",
        "label_name",
        "orig_action_name",
        "timestamp_ns",
        "t_rel_ms",
        "valid",
        "source_run",
        "source_split",
    ]

    rows_out = []
    missing_images = 0
    copy_mode_stats = defaultdict(int)
    seen_image_names = set()

    for sample in samples:
        frame = sample["frame"]
        image_name = frame["image_name"]
        if image_name in seen_image_names:
            # Prevent duplicated image rows in one run output.
            continue

        src_img_path = os.path.join(src_img_dir, image_name)
        dst_img_path = os.path.join(dst_img_dir, image_name)
        if not os.path.isfile(src_img_path):
            missing_images += 1
            continue

        mode_used = _safe_link_or_copy(src_img_path, dst_img_path, copy_mode=copy_mode)
        copy_mode_stats[mode_used] += 1
        seen_image_names.add(image_name)

        rows_out.append({
            "image_name": image_name,
            "label_id": int(sample["label_id"]),
            "label_name": sample["label_name"],
            "orig_action_name": frame["action_name"],
            "timestamp_ns": int(frame["timestamp_ns"]),
            "t_rel_ms": _format_t_rel_ms(sample["t_rel_ms"]),
            "valid": int(frame["valid"]),
            "source_run": frame["source_run"],
            "source_split": frame["source_split"],
        })

    labels_path = os.path.join(dst_run_dir, "labels.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    label_distribution = OrderedDict()
    label_distribution[LABEL_ID_TO_NAME[0]] = 0
    label_distribution[LABEL_ID_TO_NAME[1]] = 0
    for row in rows_out:
        label_distribution[row["label_name"]] += 1

    turn_summary = []
    for turn in turns:
        turn_summary.append({
            "turn_id": int(turn["turn_id"]),
            "turn_dir": turn["turn_dir"],
            "turn_on_ns": int(turn["turn_on_ns"]),
            "turn_off_ns": int(turn["turn_off_ns"]),
            "idx_on": int(turn["idx_on"]),
            "idx_off": int(turn["idx_off"]),
            "duration_ms": round((turn["turn_off_ns"] - turn["turn_on_ns"]) / 1e6, 3),
        })

    meta = OrderedDict()
    meta["dataset_name"] = DATASET_NAME
    meta["split"] = split
    meta["run_name"] = run_name
    meta["total_frames"] = len(rows_out)
    meta["label_distribution"] = label_distribution
    meta["source_type"] = source_type
    meta["derive_config"] = derive_config
    meta["detected_turn_count"] = len(turns)
    meta["turn_summary"] = turn_summary
    meta["missing_images_skipped"] = missing_images
    meta["copy_mode_stats"] = dict(copy_mode_stats)
    if extra_meta:
        meta["source_info"] = extra_meta

    with open(os.path.join(dst_run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    run_stats = {
        "split": split,
        "run_name": run_name,
        "sample_count": len(rows_out),
        "label_distribution": dict(label_distribution),
        "detected_turn_count": len(turns),
        "missing_images_skipped": missing_images,
    }
    return run_stats


def _iter_split_runs(root_dir):
    for split in SPLITS:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for run_name in sorted(os.listdir(split_dir)):
            run_dir = os.path.join(split_dir, run_name)
            if not os.path.isdir(run_dir):
                continue
            yield split, run_name, run_dir


def _init_split_summary():
    return {
        "runs_seen": 0,
        "runs_written": 0,
        "runs_skipped": 0,
        "samples_total": 0,
        "label_distribution": {
            LABEL_ID_TO_NAME[0]: 0,
            LABEL_ID_TO_NAME[1]: 0,
        },
        "runs": OrderedDict(),
    }


def _accumulate_run_summary(split_summary, run_stats):
    split_summary["runs_written"] += 1
    split_summary["samples_total"] += run_stats["sample_count"]
    split_summary["label_distribution"][LABEL_ID_TO_NAME[0]] += run_stats["label_distribution"].get(LABEL_ID_TO_NAME[0], 0)
    split_summary["label_distribution"][LABEL_ID_TO_NAME[1]] += run_stats["label_distribution"].get(LABEL_ID_TO_NAME[1], 0)
    split_summary["runs"][run_stats["run_name"]] = {
        "sample_count": run_stats["sample_count"],
        "label_distribution": run_stats["label_distribution"],
        "detected_turn_count": run_stats["detected_turn_count"],
    }


def _derive_sameenv_run(
    split,
    run_name,
    src_run_dir,
    dst_root,
    derive_config,
    args,
    source_type,
):
    labels_path = os.path.join(src_run_dir, "labels.csv")
    images_dir = os.path.join(src_run_dir, "images")
    if not os.path.isfile(labels_path) or not os.path.isdir(images_dir):
        print(f"  [Warn] {split}/{run_name}: missing labels.csv or images/, skip")
        return None, "missing_files"

    frames, skipped_rows = load_labels_csv(
        labels_path,
        valid_only=args.valid_only,
        source_split=split,
        source_run=run_name,
    )
    if skipped_rows > 0:
        print(f"  [Warn] {split}/{run_name}: skipped rows with parse errors = {skipped_rows}")
    if not frames:
        print(f"  [Warn] {split}/{run_name}: no usable frames, skip")
        return None, "no_frames"

    turns = detect_turns(frames, min_turn_k=args.min_turn_k)
    if len(turns) == 0:
        print(f"  [Warn] {split}/{run_name}: no turn detected, skip")
        return None, "no_turn"

    samples, collect_stats = collect_positive_negative_frames(
        frames=frames,
        turns=turns,
        pre_turn_ms=args.pre_turn_ms,
        pre_turn_end_ms=args.pre_turn_end_ms,
        safe_margin_ms=args.safe_margin_ms,
        post_turn_exclude_ms=args.post_turn_exclude_ms,
        stride=args.stride,
    )
    if len(samples) == 0:
        print(f"  [Warn] {split}/{run_name}: no selected samples, skip")
        return None, "no_selected_samples"

    dst_run_dir = os.path.join(dst_root, split, run_name)
    run_stats = write_derived_run(
        src_run_dir=src_run_dir,
        dst_run_dir=dst_run_dir,
        split=split,
        run_name=run_name,
        samples=samples,
        turns=turns,
        derive_config=derive_config,
        source_type=source_type,
        force=args.force,
        copy_mode=args.copy_mode,
        extra_meta={
            "source_root": os.path.abspath(args.src_root),
            "source_split": split,
            "source_run": run_name,
            "skipped_rows_in_labels": skipped_rows,
            "collect_stats": collect_stats,
            "auxiliary_straight_used": bool(args.straight_root),
        },
    )
    if run_stats is None:
        return None, "write_skip"
    return run_stats, "ok"


def _derive_aux_straight_run(
    split,
    run_name,
    src_run_dir,
    dst_root,
    derive_config,
    args,
):
    labels_path = os.path.join(src_run_dir, "labels.csv")
    images_dir = os.path.join(src_run_dir, "images")
    if not os.path.isfile(labels_path) or not os.path.isdir(images_dir):
        print(f"  [Warn] aux {split}/{run_name}: missing labels.csv or images/, skip")
        return None, "missing_files"

    frames, skipped_rows = load_labels_csv(
        labels_path,
        valid_only=args.valid_only,
        source_split=split,
        source_run=run_name,
    )
    if skipped_rows > 0:
        print(f"  [Warn] aux {split}/{run_name}: skipped rows with parse errors = {skipped_rows}")
    if not frames:
        print(f"  [Warn] aux {split}/{run_name}: no usable frames, skip")
        return None, "no_frames"

    samples = []
    for frame in frames:
        samples.append({
            "frame_idx": frame["row_idx"],
            "frame": frame,
            "label_id": 0,
            "label_name": LABEL_ID_TO_NAME[0],
            "t_rel_ms": "",
            "matched_turn_id": None,
        })

    if len(samples) == 0:
        return None, "no_selected_samples"

    out_run_name = f"{run_name}__aux_straight"
    dst_run_dir = os.path.join(dst_root, split, out_run_name)
    run_stats = write_derived_run(
        src_run_dir=src_run_dir,
        dst_run_dir=dst_run_dir,
        split=split,
        run_name=out_run_name,
        samples=samples,
        turns=[],
        derive_config=derive_config,
        source_type="mixed",
        force=args.force,
        copy_mode=args.copy_mode,
        extra_meta={
            "source_root": os.path.abspath(args.straight_root),
            "source_split": split,
            "source_run": run_name,
            "auxiliary_straight_only": True,
            "skipped_rows_in_labels": skipped_rows,
        },
    )
    if run_stats is None:
        return None, "write_skip"
    return run_stats, "ok"


def run_derive(args):
    os.makedirs(args.dst_root, exist_ok=True)
    for split in SPLITS:
        os.makedirs(os.path.join(args.dst_root, split), exist_ok=True)

    source_type = "mixed" if args.straight_root else "sameenv"
    derive_config = OrderedDict([
        ("src_root", os.path.abspath(args.src_root)),
        ("dst_root", os.path.abspath(args.dst_root)),
        ("straight_root", os.path.abspath(args.straight_root) if args.straight_root else ""),
        ("pre_turn_ms", args.pre_turn_ms),
        ("pre_turn_end_ms", args.pre_turn_end_ms),
        ("safe_margin_ms", args.safe_margin_ms),
        ("post_turn_exclude_ms", args.post_turn_exclude_ms),
        ("stride", args.stride),
        ("min_turn_k", args.min_turn_k),
        ("copy_mode", args.copy_mode),
        ("valid_only", bool(args.valid_only)),
        ("force", bool(args.force)),
        ("dataset_name", DATASET_NAME),
        ("source_type", source_type),
    ])

    summary = OrderedDict()
    summary["dataset_name"] = DATASET_NAME
    summary["source_type"] = source_type
    summary["derive_config"] = derive_config
    summary["splits"] = OrderedDict((sp, _init_split_summary()) for sp in SPLITS)
    summary["skipped_reasons"] = defaultdict(int)
    summary["auxiliary_straight_enabled"] = bool(args.straight_root)

    print("=" * 88)
    print(" Derive approach-trigger dataset")
    print("=" * 88)
    print(f" src_root      : {os.path.abspath(args.src_root)}")
    print(f" dst_root      : {os.path.abspath(args.dst_root)}")
    print(f" straight_root : {os.path.abspath(args.straight_root) if args.straight_root else '(disabled)'}")
    print(f" source_type   : {source_type}")
    print("=" * 88)

    # Primary same-environment derivation.
    if not os.path.isdir(args.src_root):
        raise FileNotFoundError(f"src_root not found: {args.src_root}")

    for split, run_name, src_run_dir in _iter_split_runs(args.src_root):
        split_summary = summary["splits"][split]
        split_summary["runs_seen"] += 1
        run_stats, status = _derive_sameenv_run(
            split=split,
            run_name=run_name,
            src_run_dir=src_run_dir,
            dst_root=args.dst_root,
            derive_config=derive_config,
            args=args,
            source_type=source_type,
        )
        if status != "ok" or run_stats is None:
            split_summary["runs_skipped"] += 1
            summary["skipped_reasons"][status] += 1
            continue
        _accumulate_run_summary(split_summary, run_stats)

    # Optional auxiliary straight negatives.
    if args.straight_root:
        if not os.path.isdir(args.straight_root):
            print(f"[Warn] straight_root not found, skip auxiliary source: {args.straight_root}")
            summary["skipped_reasons"]["straight_root_not_found"] += 1
        else:
            print("\n[Aux] Collecting auxiliary Straight negatives from straight_root ...")
            for split, run_name, src_run_dir in _iter_split_runs(args.straight_root):
                split_summary = summary["splits"][split]
                split_summary["runs_seen"] += 1
                run_stats, status = _derive_aux_straight_run(
                    split=split,
                    run_name=run_name,
                    src_run_dir=src_run_dir,
                    dst_root=args.dst_root,
                    derive_config=derive_config,
                    args=args,
                )
                if status != "ok" or run_stats is None:
                    split_summary["runs_skipped"] += 1
                    summary["skipped_reasons"][f"aux_{status}"] += 1
                    continue
                _accumulate_run_summary(split_summary, run_stats)

    # Totals.
    totals = {
        "runs_seen": 0,
        "runs_written": 0,
        "runs_skipped": 0,
        "samples_total": 0,
        "label_distribution": {
            LABEL_ID_TO_NAME[0]: 0,
            LABEL_ID_TO_NAME[1]: 0,
        },
    }
    for split in SPLITS:
        ss = summary["splits"][split]
        totals["runs_seen"] += ss["runs_seen"]
        totals["runs_written"] += ss["runs_written"]
        totals["runs_skipped"] += ss["runs_skipped"]
        totals["samples_total"] += ss["samples_total"]
        totals["label_distribution"][LABEL_ID_TO_NAME[0]] += ss["label_distribution"][LABEL_ID_TO_NAME[0]]
        totals["label_distribution"][LABEL_ID_TO_NAME[1]] += ss["label_distribution"][LABEL_ID_TO_NAME[1]]
    summary["totals"] = totals
    summary["skipped_reasons"] = dict(summary["skipped_reasons"])

    # Write derive_summary.json
    summary_path = os.path.join(args.dst_root, "derive_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Terminal summary.
    print("\n" + "=" * 88)
    print(" Derive Summary")
    print("=" * 88)
    for split in SPLITS:
        ss = summary["splits"][split]
        print(f"[{split}]")
        print(f"  runs_seen     : {ss['runs_seen']}")
        print(f"  runs_written  : {ss['runs_written']}")
        print(f"  runs_skipped  : {ss['runs_skipped']}")
        print(f"  samples_total : {ss['samples_total']}")
        print(
            "  label_dist    : "
            f"{LABEL_ID_TO_NAME[0]}={ss['label_distribution'][LABEL_ID_TO_NAME[0]]}, "
            f"{LABEL_ID_TO_NAME[1]}={ss['label_distribution'][LABEL_ID_TO_NAME[1]]}"
        )
        if ss["runs"]:
            print("  per_run:")
            for rn, rst in ss["runs"].items():
                print(
                    f"    - {rn}: total={rst['sample_count']}, "
                    f"{LABEL_ID_TO_NAME[0]}={rst['label_distribution'].get(LABEL_ID_TO_NAME[0], 0)}, "
                    f"{LABEL_ID_TO_NAME[1]}={rst['label_distribution'].get(LABEL_ID_TO_NAME[1], 0)}"
                )
    print("-" * 88)
    print(f"totals.runs_seen     : {totals['runs_seen']}")
    print(f"totals.runs_written  : {totals['runs_written']}")
    print(f"totals.runs_skipped  : {totals['runs_skipped']}")
    print(f"totals.samples_total : {totals['samples_total']}")
    print(
        "totals.label_dist    : "
        f"{LABEL_ID_TO_NAME[0]}={totals['label_distribution'][LABEL_ID_TO_NAME[0]]}, "
        f"{LABEL_ID_TO_NAME[1]}={totals['label_distribution'][LABEL_ID_TO_NAME[1]]}"
    )
    if summary["skipped_reasons"]:
        print(f"skipped_reasons      : {summary['skipped_reasons']}")
    print(f"[OK] {summary_path}")
    print("=" * 88)


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Derive approach-trigger dataset from corridor_task runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default="./data/corridor_stage3_rawsplit",
        help="Primary source root: src_root/{train,val,test}/{run_name}/",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default=f"./data/stage1_v3/{DATASET_NAME}",
        help="Destination dataset root.",
    )
    parser.add_argument(
        "--straight_root",
        type=str,
        default="",
        help="Optional auxiliary straight source root (disabled when empty).",
    )
    parser.add_argument("--pre_turn_ms", type=int, default=700)
    parser.add_argument("--pre_turn_end_ms", type=int, default=100)
    parser.add_argument("--safe_margin_ms", type=int, default=2500)
    parser.add_argument("--post_turn_exclude_ms", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--min_turn_k", type=int, default=3)
    parser.add_argument(
        "--copy_mode",
        type=str,
        default="symlink",
        choices=["copy", "symlink"],
        help="How to materialize image files in derived dataset.",
    )
    parser.add_argument(
        "--valid_only",
        type=str2bool,
        default=True,
        help="Keep only valid=1 frames from source labels.csv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination run directories.",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.pre_turn_ms < args.pre_turn_end_ms:
        raise ValueError("--pre_turn_ms must be >= --pre_turn_end_ms")
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")
    if args.min_turn_k <= 0:
        raise ValueError("--min_turn_k must be >= 1")

    run_derive(args)


if __name__ == "__main__":
    main()
