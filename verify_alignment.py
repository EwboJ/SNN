#!/usr/bin/env python3
"""
verify_alignment.py — 验证 corridor_export.py 导出数据的对齐正确性
================================================================

功能:
  - 随机抽取 N 帧（默认 50），可视化显示:
      图像 + cmd_vel (linear_x, angular_z) + 离散动作标签 + 对齐时间差
  - 生成拼接大图保存 或 逐帧交互查看
  - 输出对齐质量汇总统计

依赖:
  pip install numpy opencv-python matplotlib pandas

使用:
  # 随机可视化 50 帧（保存为拼图）
  python verify_alignment.py --data ./data/corridor_run1

  # 交互模式逐帧查看
  python verify_alignment.py --data ./data/corridor_run1 --interactive

  # 指定抽样数量
  python verify_alignment.py --data ./data/corridor_run1 --num 100

  # 按时间差排序（检查对齐最差的帧）
  python verify_alignment.py --data ./data/corridor_run1 --sort-by-diff --num 30
"""

import os
import sys
import argparse
import json

import numpy as np
import pandas as pd
import cv2

try:
    import matplotlib
    matplotlib.use("Agg")  # 默认非交互后端，交互模式会切换
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# 动作标签颜色 (BGR for OpenCV, RGB for matplotlib)
ACTION_COLORS_BGR = {
    "Forward":  (0, 200, 0),     # 绿
    "Backward": (0, 0, 200),     # 红
    "Left":     (200, 200, 0),   # 青
    "Right":    (0, 165, 255),   # 橙
    "Stop":     (128, 128, 128), # 灰
    "invalid":  (0, 0, 255),     # 红
}

ACTION_COLORS_RGB = {
    "Forward":  "#00C800",
    "Backward": "#C80000",
    "Left":     "#00C8C8",
    "Right":    "#FFA500",
    "Stop":     "#808080",
    "invalid":  "#FF0000",
}


def load_dataset(data_dir: str) -> pd.DataFrame:
    """加载 labels.csv"""
    csv_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] 未找到 labels.csv: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print(f"[INFO] 加载 {len(df)} 条记录 from {csv_path}")
    return df


def print_summary(df: pd.DataFrame, data_dir: str):
    """打印对齐质量汇总"""
    print("\n" + "=" * 50)
    print("  对齐质量汇总")
    print("=" * 50)

    total = len(df)
    valid = df["valid"].astype(int).sum()
    invalid = total - valid
    print(f"  总帧数: {total}")
    print(f"  有效帧: {valid} ({100 * valid / total:.1f}%)")
    print(f"  无效帧: {invalid} ({100 * invalid / total:.1f}%)")

    if "time_diff_ms" in df.columns:
        valid_diffs = df.loc[df["valid"].astype(int) == 1, "time_diff_ms"]
        if len(valid_diffs) > 0:
            print(f"\n  时间差统计 (有效帧):")
            print(f"    均值:  {valid_diffs.mean():.2f} ms")
            print(f"    中位数: {valid_diffs.median():.2f} ms")
            print(f"    最大:  {valid_diffs.max():.2f} ms")
            print(f"    P95:   {np.percentile(valid_diffs, 95):.2f} ms")
            print(f"    P99:   {np.percentile(valid_diffs, 99):.2f} ms")

    print(f"\n  动作分布:")
    action_counts = df["action_name"].value_counts()
    for action, count in action_counts.items():
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"    {action:>10s}: {count:5d} ({pct:5.1f}%) {bar}")

    # 加载 meta.json
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        print(f"\n  元数据:")
        for key in ["corridor_width_m", "lighting", "robot_id",
                     "camera_height_m", "duration_seconds"]:
            if key in meta:
                print(f"    {key}: {meta[key]}")

    print("=" * 50)


def visualize_grid_matplotlib(df: pd.DataFrame, data_dir: str,
                              num: int, sort_by_diff: bool,
                              output_path: str):
    """用 matplotlib 生成 grid 可视化大图"""
    if not HAS_MPL:
        print("[ERROR] matplotlib 未安装，请运行: pip install matplotlib")
        return

    images_dir = os.path.join(data_dir, "images")

    # 选择样本
    if sort_by_diff and "time_diff_ms" in df.columns:
        sample_df = df.nlargest(num, "time_diff_ms")
        print(f"[INFO] 按时间差降序选取 top-{num} 帧 (检查最差对齐)")
    else:
        n = min(num, len(df))
        sample_df = df.sample(n=n, random_state=42)
        print(f"[INFO] 随机抽取 {n} 帧")

    sample_df = sample_df.sort_values("timestamp_ns").reset_index(drop=True)

    # 计算网格布局
    n = len(sample_df)
    cols = min(10, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 2.5, rows * 3.0), dpi=100)
    fig.suptitle(f"对齐验证 — {data_dir}\n"
                 f"共 {len(df)} 帧, 显示 {n} 帧",
                 fontsize=14, fontweight='bold')

    gs = GridSpec(rows, cols, figure=fig, hspace=0.5, wspace=0.15)

    for i, (_, row) in enumerate(sample_df.iterrows()):
        ax = fig.add_subplot(gs[i // cols, i % cols])

        img_path = os.path.join(images_dir, row["image_name"])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "MISSING", ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='red')

        action = row["action_name"]
        color = ACTION_COLORS_RGB.get(action, "#000000")
        lx = row.get("linear_x", 0)
        az = row.get("angular_z", 0)
        td = row.get("time_diff_ms", -1)
        valid = bool(int(row.get("valid", 0)))

        title = f"{action}"
        if valid:
            title += f"\nlx={lx:.3f} az={az:.3f}"
            title += f"\nΔt={td:.1f}ms"
        else:
            title += f"\n[INVALID Δt={td:.1f}ms]"

        ax.set_title(title, fontsize=7, color=color, fontweight='bold')
        ax.axis("off")

        # 边框颜色
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    # 隐藏多余的子图
    for j in range(n, rows * cols):
        ax = fig.add_subplot(gs[j // cols, j % cols])
        ax.axis("off")

    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    print(f"[INFO] 可视化大图已保存 → {output_path}")
    plt.close(fig)


def visualize_interactive(df: pd.DataFrame, data_dir: str,
                          num: int, sort_by_diff: bool):
    """用 OpenCV 逐帧交互查看"""
    images_dir = os.path.join(data_dir, "images")

    if sort_by_diff and "time_diff_ms" in df.columns:
        sample_df = df.nlargest(num, "time_diff_ms")
    else:
        n = min(num, len(df))
        sample_df = df.sample(n=n, random_state=42)

    sample_df = sample_df.sort_values("timestamp_ns").reset_index(drop=True)
    n = len(sample_df)

    print(f"\n[交互模式] 共 {n} 帧")
    print("  按键: →/D 下一帧   ←/A 上一帧   Q 退出")
    print("=" * 40)

    idx = 0
    while True:
        row = sample_df.iloc[idx]
        img_path = os.path.join(images_dir, row["image_name"])

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "IMAGE MISSING", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 绘制信息叠加
        action = row["action_name"]
        color = ACTION_COLORS_BGR.get(action, (255, 255, 255))
        lx = row.get("linear_x", 0)
        az = row.get("angular_z", 0)
        td = row.get("time_diff_ms", -1)
        valid = bool(int(row.get("valid", 0)))

        h, w = img.shape[:2]

        # 半透明信息面板
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # 文字信息
        cv2.putText(img, f"[{idx + 1}/{n}] {row['image_name']}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f"Action: {action}",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        vel_text = f"linear_x={lx:.4f}  angular_z={az:.4f}"
        cv2.putText(img, vel_text,
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        valid_text = f"dt={td:.1f}ms  {'VALID' if valid else 'INVALID'}"
        valid_color = (0, 255, 0) if valid else (0, 0, 255)
        cv2.putText(img, valid_text,
                    (w - 250, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, valid_color, 1)

        # 速度方向箭头
        cx, cy = w // 2, h - 60
        # 前进/后退箭头
        arrow_len = int(min(abs(lx) * 300, 80))
        if lx > 0:
            cv2.arrowedLine(img, (cx, cy), (cx, cy - arrow_len),
                            (0, 255, 0), 3, tipLength=0.3)
        elif lx < 0:
            cv2.arrowedLine(img, (cx, cy), (cx, cy + arrow_len),
                            (0, 0, 255), 3, tipLength=0.3)
        # 转向箭头
        turn_len = int(min(abs(az) * 150, 80))
        if az > 0:   # 左转
            cv2.arrowedLine(img, (cx, cy), (cx - turn_len, cy),
                            (255, 255, 0), 3, tipLength=0.3)
        elif az < 0:  # 右转
            cv2.arrowedLine(img, (cx, cy), (cx + turn_len, cy),
                            (0, 165, 255), 3, tipLength=0.3)

        cv2.imshow("Alignment Verification", img)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), ord('Q'), 27):    # Q or ESC
            break
        elif key in (ord('d'), ord('D'), 83):  # → or D
            idx = min(idx + 1, n - 1)
        elif key in (ord('a'), ord('A'), 81):  # ← or A
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()


def visualize_cmd_vel_timeline(df: pd.DataFrame, data_dir: str, output_path: str):
    """绘制 cmd_vel 时间线图，用于全局检查"""
    if not HAS_MPL:
        return

    valid_df = df[df["valid"].astype(int) == 1].copy()
    if len(valid_df) == 0:
        print("[WARNING] 无有效帧，跳过时间线绘制。")
        return

    t0 = valid_df["timestamp_ns"].min()
    valid_df["time_s"] = (valid_df["timestamp_ns"] - t0) / 1e9

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"cmd_vel 时间线 — {data_dir}", fontsize=13, fontweight='bold')

    # 1) linear_x
    axes[0].plot(valid_df["time_s"], valid_df["linear_x"],
                 'b-', linewidth=0.8, alpha=0.8)
    axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].set_ylabel("linear_x (m/s)")
    axes[0].grid(True, alpha=0.3)

    # 2) angular_z
    axes[1].plot(valid_df["time_s"], valid_df["angular_z"],
                 'r-', linewidth=0.8, alpha=0.8)
    axes[1].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[1].set_ylabel("angular_z (rad/s)")
    axes[1].grid(True, alpha=0.3)

    # 3) 动作标签散点
    for action, color in ACTION_COLORS_RGB.items():
        mask = valid_df["action_name"] == action
        if mask.any():
            axes[2].scatter(valid_df.loc[mask, "time_s"],
                            valid_df.loc[mask, "action_id"],
                            c=color, label=action, s=8, alpha=0.7)
    axes[2].set_ylabel("action_id")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc='upper right', fontsize=8, markerscale=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"[INFO] 时间线图已保存 → {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="验证 corridor_export.py 导出数据的对齐正确性",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--data", type=str, required=True,
                        help="导出数据目录 (包含 images/ 和 labels.csv)")
    parser.add_argument("--num", type=int, default=50,
                        help="抽样帧数 (默认: 50)")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式 (用 OpenCV 逐帧查看)")
    parser.add_argument("--sort-by-diff", action="store_true",
                        help="按时间差降序选取 (检查对齐最差的帧)")
    parser.add_argument("--no-timeline", action="store_true",
                        help="不绘制 cmd_vel 时间线图")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="输出文件名前缀 (默认: data目录下)")

    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"[ERROR] 数据目录不存在: {args.data}")
        sys.exit(1)

    # 加载数据
    df = load_dataset(args.data)

    # 汇总统计
    print_summary(df, args.data)

    # 输出路径
    prefix = args.output_prefix or os.path.join(args.data, "verify")

    if args.interactive:
        # 交互模式
        matplotlib.use("TkAgg")
        visualize_interactive(df, args.data, args.num, args.sort_by_diff)
    else:
        # 保存网格大图
        grid_path = f"{prefix}_grid_{args.num}.png"
        visualize_grid_matplotlib(df, args.data, args.num,
                                  args.sort_by_diff, grid_path)

    # 时间线图
    if not args.no_timeline:
        timeline_path = f"{prefix}_timeline.png"
        visualize_cmd_vel_timeline(df, args.data, timeline_path)

    print("\n[DONE] 验证完成。")


if __name__ == "__main__":
    main()
