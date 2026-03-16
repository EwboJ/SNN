#!/usr/bin/env python3
"""
corridor_export.py — 从 ROS2 rosbag2 导出走廊导航数据集
=========================================================
# -------- 终端 3：录制 rosbag2 --------
ros2 bag record /camera/image_raw /cmd_vel /odom_raw -o corridor_bag1

功能:
  1) 从 rosbag2 (.db3) 按时间戳排序导出图像帧 (jpg/png)
  2) 每帧匹配最近的 /cmd_vel，支持最近邻 / 线性插值 / 最大允许时间差
  3) 将 cmd_vel 离散化为动作标签 (Forward/Left/Right/Backward/Stop)
  4) 输出 labels.csv + meta.json
  5) 可选导出 /odom_raw → odom_raw.csv (用于直行纠偏分析)

依赖:
  pip install rosbags numpy opencv-python pyyaml

使用:
  python corridor_export.py --bag ./nav --output ./data/corridor_run1
  python corridor_export.py --bag ./nav --output ./data/corridor_run1 --config corridor_config.yaml

输出结构:
  output_dir/
    ├── images/            # 按时间戳排序的图像帧
    ├── labels.csv         # image_name, action_id, action_name, timestamp_ns,
    │                      #   linear_x, angular_z, time_diff_ms, valid
    ├── odom_raw.csv       # (可选) timestamp_ns, x, y, yaw, linear_v, angular_w
    └── meta.json          # 元数据 + 配置快照 + 统计信息
"""

import os
import sys
import csv
import json
import argparse
import math
import bisect
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import yaml
import numpy as np
import cv2

# ---- rosbags (纯 Python ROS2 bag 读取库) ----
try:
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr
    HAS_ROSBAGS = True
except ImportError:
    HAS_ROSBAGS = False


# ==============================================================
#  默认配置（如果未指定 yaml 则使用）
# ==============================================================
DEFAULT_CONFIG = {
    "bag": {
        "image_topic": "/camera/color/image_raw",
        "cmd_vel_topic": "/cmd_vel",
        "odom_topic": "/odom_raw",
        "image_encoding": "auto",
    },
    "image": {
        "format": "jpg",
        "jpg_quality": 95,
        "resize": None,
        "sample_interval_ms": 0,
    },
    "alignment": {
        "method": "nearest",          # nearest / linear_interp
        "max_time_diff_ms": 100,
    },
    "action_thresholds": {
        "forward":  {"linear_x_min": 0.05,  "angular_z_max": 0.3},
        "backward": {"linear_x_max": -0.05, "angular_z_max": 0.3},
        "left":     {"angular_z_min": 0.3},
        "right":    {"angular_z_max": -0.3},
        "stop":     {"enabled": True, "linear_x_abs_max": 0.05, "angular_z_abs_max": 0.3},
    },
    "action_ids": {
        "Forward": 0, "Backward": 1, "Left": 2, "Right": 3, "Stop": 4,
    },
    "meta": {
        "corridor_width_m": 2.0,
        "lighting": "fluorescent",
        "robot_id": "turtlebot3_burger",
        "camera_height_m": 0.15,
        "camera_fov_deg": 60,
        "camera_resolution": [640, 480],
        "location": "indoor_corridor",
        "notes": "",
    },
}


def deep_update(base: dict, override: dict) -> dict:
    """递归更新嵌套字典"""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: Optional[str]) -> dict:
    """加载 YAML 配置，合并到默认配置"""
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path and os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_cfg = yaml.safe_load(f) or {}
        deep_update(cfg, user_cfg)
        print(f"[INFO] 已加载配置: {config_path}")
    else:
        print(f"[INFO] 使用默认配置")
    return cfg


# ==============================================================
#  从 ROS2 bag 读取消息
# ==============================================================
def read_bag_messages(bag_path: str, image_topic: str, cmd_vel_topic: str,
                      odom_topic: str = "/odom_raw"):
    """
    读取 rosbag2，返回:
      image_msgs: list of (timestamp_ns, image_ndarray)
      cmd_vel_msgs: list of (timestamp_ns, linear_x, linear_y, angular_z)
      odom_msgs: list of (timestamp_ns, x, y, yaw, linear_v, angular_w)

    所有列表均按 timestamp_ns 升序排列。
    若 bag 中无 odom topic，odom_msgs 返回空列表（仅 warning）。
    """
    if not HAS_ROSBAGS:
        print("[ERROR] 未安装 rosbags，请运行: pip install rosbags")
        sys.exit(1)

    image_msgs = []
    cmd_vel_msgs = []
    odom_msgs = []

    with Reader(bag_path) as reader:
        # 打印 bag 中所有话题
        print(f"\n[INFO] Bag 话题列表:")
        topic_set = set()
        for conn in reader.connections:
            if conn.topic not in topic_set:
                print(f"  {conn.topic}  [{conn.msgtype}]")
                topic_set.add(conn.topic)

        # 筛选连接
        img_conns = [c for c in reader.connections if c.topic == image_topic]
        vel_conns = [c for c in reader.connections if c.topic == cmd_vel_topic]
        odom_conns = [c for c in reader.connections if c.topic == odom_topic]

        if not img_conns:
            print(f"[ERROR] 未找到图像话题: {image_topic}")
            print(f"  可用话题: {[c.topic for c in reader.connections]}")
            sys.exit(1)
        if not vel_conns:
            print(f"[WARNING] 未找到速度话题: {cmd_vel_topic}")
            print(f"  将只导出图像，所有帧标记为 unlabeled。")
        if not odom_conns:
            print(f"[WARNING] 未找到里程计话题: {odom_topic}")
            print(f"  将跳过 odom_raw.csv 导出。")

        all_conns = img_conns + vel_conns + odom_conns

        for conn, timestamp, rawdata in reader.messages(connections=all_conns):
            msg = deserialize_cdr(rawdata, conn.msgtype)

            if conn.topic == image_topic:
                img = _decode_image(msg)
                if img is not None:
                    image_msgs.append((timestamp, img))

            elif conn.topic == cmd_vel_topic:
                lx = float(msg.linear.x)
                ly = float(msg.linear.y)
                az = float(msg.angular.z)
                cmd_vel_msgs.append((timestamp, lx, ly, az))

            elif conn.topic == odom_topic:
                odom_data = _extract_odom(msg, timestamp)
                if odom_data is not None:
                    odom_msgs.append(odom_data)

    # 按时间戳排序
    image_msgs.sort(key=lambda x: x[0])
    cmd_vel_msgs.sort(key=lambda x: x[0])
    odom_msgs.sort(key=lambda x: x[0])

    print(f"\n[INFO] 图像帧数: {len(image_msgs)}")
    print(f"[INFO] cmd_vel 消息数: {len(cmd_vel_msgs)}")
    print(f"[INFO] odom 消息数: {len(odom_msgs)}")

    if image_msgs:
        dt = (image_msgs[-1][0] - image_msgs[0][0]) / 1e9
        print(f"[INFO] 图像时间跨度: {dt:.2f} 秒")
        if len(image_msgs) > 1:
            avg_fps = (len(image_msgs) - 1) / dt if dt > 0 else 0
            print(f"[INFO] 平均帧率: {avg_fps:.1f} FPS")

    return image_msgs, cmd_vel_msgs, odom_msgs


def _extract_odom(msg, timestamp: int) -> Optional[tuple]:
    """
    从 nav_msgs/Odometry 消息中提取 (timestamp_ns, x, y, yaw, linear_v, angular_w)。

    支持:
      - nav_msgs/msg/Odometry (标准)
      - geometry_msgs/msg/PoseStamped (仅位姿, 速度设为 0)
    """
    try:
        # nav_msgs/Odometry
        pose = msg.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)

        # 四元数 → yaw
        q = pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

        # 速度
        try:
            twist = msg.twist.twist
            linear_v = float(twist.linear.x)
            angular_w = float(twist.angular.z)
        except AttributeError:
            linear_v = 0.0
            angular_w = 0.0

        return (timestamp, x, y, yaw, linear_v, angular_w)
    except AttributeError:
        # 非标准消息格式
        pass

    try:
        # geometry_msgs/PoseStamped
        pose = msg.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        q = pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        return (timestamp, x, y, yaw, 0.0, 0.0)
    except AttributeError:
        pass

    print(f"[WARNING] odom 消息解析失败 (ts={timestamp})")
    return None


def _decode_image(msg) -> Optional[np.ndarray]:
    """将 ROS Image 消息解码为 BGR ndarray"""
    try:
        h, w = msg.height, msg.width
        encoding = getattr(msg, 'encoding', 'bgr8')

        if encoding in ('rgb8', 'bgr8'):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            if encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif encoding in ('mono8', '8UC1'):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif encoding in ('16UC1', 'mono16'):
            img = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
            img = (img / 256).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif encoding == 'bgra8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif encoding == 'rgba8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            # 尝试按 3 通道解码
            step = getattr(msg, 'step', w * 3)
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, step)
            img = img[:, :w * 3].reshape(h, w, 3)
        return img.copy()   # copy 防止 buffer 被回收
    except Exception as e:
        print(f"[WARNING] 图像解码失败: {e}")
        return None


# ==============================================================
#  cmd_vel 对齐
# ==============================================================
def align_cmd_vel(image_timestamps: List[int],
                  cmd_vel_msgs: List[tuple],
                  method: str = "nearest",
                  max_diff_ms: float = 100) -> List[dict]:
    """
    对齐图像帧的 cmd_vel。

    Args:
        image_timestamps: 图像时间戳列表 (ns)
        cmd_vel_msgs: [(ts_ns, lx, ly, az), ...]
        method: "nearest" 或 "linear_interp"
        max_diff_ms: 最大允许时间差 (ms)

    Returns:
        list of dict: 每帧对应一个 {linear_x, linear_y, angular_z, time_diff_ms, valid}
    """
    if not cmd_vel_msgs:
        return [{"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0,
                 "time_diff_ms": float('inf'), "valid": False}
                for _ in image_timestamps]

    vel_ts = [m[0] for m in cmd_vel_msgs]
    max_diff_ns = max_diff_ms * 1e6

    results = []

    for img_ts in image_timestamps:
        idx = bisect.bisect_left(vel_ts, img_ts)

        if method == "nearest":
            result = _nearest_match(img_ts, idx, cmd_vel_msgs, vel_ts, max_diff_ns)
        elif method == "linear_interp":
            result = _linear_interp_match(img_ts, idx, cmd_vel_msgs, vel_ts, max_diff_ns)
        else:
            raise ValueError(f"未知对齐方法: {method}")

        results.append(result)

    return results


def _nearest_match(img_ts, idx, cmd_vel_msgs, vel_ts, max_diff_ns) -> dict:
    """最近邻匹配"""
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(vel_ts):
        candidates.append(idx)

    best_i = None
    best_diff = float('inf')
    for ci in candidates:
        diff = abs(img_ts - vel_ts[ci])
        if diff < best_diff:
            best_diff = diff
            best_i = ci

    if best_i is None:
        return {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0,
                "time_diff_ms": float('inf'), "valid": False}

    valid = best_diff <= max_diff_ns
    _, lx, ly, az = cmd_vel_msgs[best_i]
    return {"linear_x": lx, "linear_y": ly, "angular_z": az,
            "time_diff_ms": best_diff / 1e6, "valid": valid}


def _linear_interp_match(img_ts, idx, cmd_vel_msgs, vel_ts, max_diff_ns) -> dict:
    """线性插值匹配：在前后两个 cmd_vel 消息之间插值"""
    # 边界情况：img_ts 在所有 vel 之前或之后
    if idx == 0:
        return _nearest_match(img_ts, idx, cmd_vel_msgs, vel_ts, max_diff_ns)
    if idx >= len(vel_ts):
        return _nearest_match(img_ts, idx, cmd_vel_msgs, vel_ts, max_diff_ns)

    ts_before = vel_ts[idx - 1]
    ts_after = vel_ts[idx]

    # 检查两端是否都在容许范围内
    diff_before = img_ts - ts_before
    diff_after = ts_after - img_ts

    if diff_before > max_diff_ns and diff_after > max_diff_ns:
        return {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0,
                "time_diff_ms": min(diff_before, diff_after) / 1e6, "valid": False}

    # 线性插值
    span = ts_after - ts_before
    if span == 0:
        alpha = 0.5
    else:
        alpha = (img_ts - ts_before) / span   # 0~1

    _, lx0, ly0, az0 = cmd_vel_msgs[idx - 1]
    _, lx1, ly1, az1 = cmd_vel_msgs[idx]

    lx = lx0 + alpha * (lx1 - lx0)
    ly = ly0 + alpha * (ly1 - ly0)
    az = az0 + alpha * (az1 - az0)

    time_diff = min(diff_before, diff_after) / 1e6
    return {"linear_x": lx, "linear_y": ly, "angular_z": az,
            "time_diff_ms": time_diff, "valid": True}


# ==============================================================
#  动作离散化
# ==============================================================
def discretize_action(linear_x: float, angular_z: float,
                      thresholds: dict, action_ids: dict) -> Tuple[int, str]:
    """
    将连续 cmd_vel 离散化为动作标签。

    优先级: Stop → Left → Right → Forward → Backward → Stop(fallback)
    """
    th = thresholds

    # 1) Stop 判断（如果启用）
    stop_cfg = th.get("stop", {})
    if stop_cfg.get("enabled", False):
        lx_abs_max = stop_cfg.get("linear_x_abs_max", 0.05)
        az_abs_max = stop_cfg.get("angular_z_abs_max", 0.3)
        if abs(linear_x) < lx_abs_max and abs(angular_z) < az_abs_max:
            return action_ids.get("Stop", 4), "Stop"

    # 2) Left
    left_cfg = th.get("left", {})
    az_min_left = left_cfg.get("angular_z_min", 0.3)
    if angular_z >= az_min_left:
        return action_ids.get("Left", 2), "Left"

    # 3) Right
    right_cfg = th.get("right", {})
    az_max_right = right_cfg.get("angular_z_max", -0.3)
    if angular_z <= az_max_right:
        return action_ids.get("Right", 3), "Right"

    # 4) Forward
    fwd_cfg = th.get("forward", {})
    lx_min_fwd = fwd_cfg.get("linear_x_min", 0.05)
    az_max_fwd = fwd_cfg.get("angular_z_max", 0.3)
    if linear_x >= lx_min_fwd and abs(angular_z) < az_max_fwd:
        return action_ids.get("Forward", 0), "Forward"

    # 5) Backward
    bwd_cfg = th.get("backward", {})
    lx_max_bwd = bwd_cfg.get("linear_x_max", -0.05)
    az_max_bwd = bwd_cfg.get("angular_z_max", 0.3)
    if linear_x <= lx_max_bwd and abs(angular_z) < az_max_bwd:
        return action_ids.get("Backward", 1), "Backward"

    # 6) 兜底: Stop（速度很小但不严格为零的情况）
    if stop_cfg.get("enabled", False):
        return action_ids.get("Stop", 4), "Stop"

    # Stop 未启用时兜底为 Forward
    return action_ids.get("Forward", 0), "Forward"


# ==============================================================
#  保存图像帧
# ==============================================================
def save_frames(image_msgs: list, output_dir: str, cfg: dict,
                sample_interval_ms: int = 0) -> List[tuple]:
    """
    保存图像并返回 (image_name, timestamp_ns) 列表。
    如果设置了 sample_interval_ms > 0，则按间隔跳帧。
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    img_format = cfg["image"]["format"].lower()
    jpg_quality = cfg["image"].get("jpg_quality", 95)
    resize = cfg["image"].get("resize", None)
    interval_ns = sample_interval_ms * 1e6

    saved = []
    last_ts = -float('inf')

    for i, (ts, img) in enumerate(image_msgs):
        # 跳帧
        if interval_ns > 0 and (ts - last_ts) < interval_ns:
            continue
        last_ts = ts

        # 缩放
        if resize and isinstance(resize, (list, tuple)) and len(resize) == 2:
            img = cv2.resize(img, tuple(resize))

        idx = len(saved)
        if img_format == "png":
            name = f"{idx:06d}.png"
            cv2.imwrite(os.path.join(images_dir, name), img)
        else:
            name = f"{idx:06d}.jpg"
            cv2.imwrite(os.path.join(images_dir, name), img,
                        [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

        saved.append((name, ts))

        if (idx + 1) % 200 == 0:
            print(f"  已保存 {idx + 1} 帧...")

    print(f"[INFO] 共保存 {len(saved)} 帧 → {images_dir}")
    return saved


# ==============================================================
#  写入 labels.csv
# ==============================================================
def write_labels_csv(output_dir: str, saved_frames: List[tuple],
                     aligned: List[dict], cfg: dict) -> dict:
    """
    写入 labels.csv 并返回统计信息。

    CSV 列: image_name, action_id, action_name, timestamp_ns,
            linear_x, angular_z, time_diff_ms, valid
    """
    thresholds = cfg["action_thresholds"]
    action_ids = cfg["action_ids"]

    csv_path = os.path.join(output_dir, "labels.csv")
    stats = {"total": 0, "valid": 0, "invalid": 0, "action_counts": {}}

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "action_id", "action_name",
                         "timestamp_ns", "linear_x", "angular_z",
                         "time_diff_ms", "valid"])

        for (img_name, ts_ns), vel in zip(saved_frames, aligned):
            lx = vel["linear_x"]
            az = vel["angular_z"]
            td = vel["time_diff_ms"]
            valid = vel["valid"]

            if valid:
                aid, aname = discretize_action(lx, az, thresholds, action_ids)
            else:
                aid, aname = -1, "invalid"

            writer.writerow([img_name, aid, aname, ts_ns,
                             f"{lx:.6f}", f"{az:.6f}",
                             f"{td:.3f}", int(valid)])

            stats["total"] += 1
            if valid:
                stats["valid"] += 1
                stats["action_counts"][aname] = stats["action_counts"].get(aname, 0) + 1
            else:
                stats["invalid"] += 1

    print(f"[INFO] labels.csv → {csv_path}")
    print(f"[INFO] 有效帧: {stats['valid']} / {stats['total']}  "
          f"(无效: {stats['invalid']})")
    print(f"[INFO] 动作分布: {stats['action_counts']}")
    return stats


# ==============================================================
#  写入 meta.json
# ==============================================================
def write_odom_csv(output_dir: str, odom_msgs: List[tuple]) -> int:
    """
    写入 odom_raw.csv。

    CSV 列: timestamp_ns, x, y, yaw, linear_v, angular_w

    Returns:
        int: 写入行数
    """
    if not odom_msgs:
        print("[INFO] 无 odom 数据，跳过 odom_raw.csv")
        return 0

    csv_path = os.path.join(output_dir, "odom_raw.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "x", "y", "yaw",
                         "linear_v", "angular_w"])
        for (ts, x, y, yaw, lv, aw) in odom_msgs:
            writer.writerow([ts, f"{x:.6f}", f"{y:.6f}", f"{yaw:.6f}",
                             f"{lv:.6f}", f"{aw:.6f}"])

    print(f"[INFO] odom_raw.csv → {csv_path}  ({len(odom_msgs)} 条)")
    return len(odom_msgs)


def write_meta_json(output_dir: str, cfg: dict, stats: dict,
                    bag_path: str, saved_frames: List[tuple],
                    odom_count: int = 0):
    """写入 meta.json，包含配置快照 + 统计信息 + odom 信息"""
    meta = dict(cfg["meta"])
    meta["export_time"] = datetime.now().isoformat()
    meta["bag_path"] = os.path.abspath(bag_path)
    meta["total_frames"] = stats["total"]
    meta["valid_frames"] = stats["valid"]
    meta["invalid_frames"] = stats["invalid"]
    meta["action_distribution"] = stats["action_counts"]

    if saved_frames:
        meta["first_timestamp_ns"] = int(saved_frames[0][1])
        meta["last_timestamp_ns"] = int(saved_frames[-1][1])
        duration_s = (saved_frames[-1][1] - saved_frames[0][1]) / 1e9
        meta["duration_seconds"] = round(duration_s, 3)
    else:
        meta["duration_seconds"] = 0

    # odom 信息
    odom_topic = cfg["bag"].get("odom_topic", "/odom_raw")
    meta["odom_available"] = odom_count > 0
    meta["odom_topic"] = odom_topic
    meta["odom_count"] = odom_count

    # 配置快照
    meta["config_snapshot"] = {
        "image_topic": cfg["bag"]["image_topic"],
        "cmd_vel_topic": cfg["bag"]["cmd_vel_topic"],
        "odom_topic": odom_topic,
        "alignment_method": cfg["alignment"]["method"],
        "max_time_diff_ms": cfg["alignment"]["max_time_diff_ms"],
        "action_thresholds": cfg["action_thresholds"],
        "action_ids": cfg["action_ids"],
        "image_format": cfg["image"]["format"],
        "resize": cfg["image"]["resize"],
        "sample_interval_ms": cfg["image"]["sample_interval_ms"],
    }

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[INFO] meta.json → {meta_path}")


# ==============================================================
#  主流程
# ==============================================================
def run_export(bag_path: str, output_dir: str, cfg: dict):
    """完整导出流程"""
    print("=" * 60)
    print("  corridor_export — ROS2 走廊数据集导出工具")
    print("=" * 60)

    # 1) 读取 bag
    image_topic = cfg["bag"]["image_topic"]
    cmd_vel_topic = cfg["bag"]["cmd_vel_topic"]
    odom_topic = cfg["bag"].get("odom_topic", "/odom_raw")
    image_msgs, cmd_vel_msgs, odom_msgs = read_bag_messages(
        bag_path, image_topic, cmd_vel_topic, odom_topic)

    if not image_msgs:
        print("[ERROR] 未读取到任何图像帧，请检查话题名称。")
        sys.exit(1)

    # 2) 保存图像
    sample_ms = cfg["image"].get("sample_interval_ms", 0)
    saved_frames = save_frames(image_msgs, output_dir, cfg, sample_ms)

    if not saved_frames:
        print("[ERROR] 未保存任何帧（可能采样间隔过大）。")
        sys.exit(1)

    # 3) 对齐 cmd_vel
    image_timestamps = [ts for (_, ts) in saved_frames]
    method = cfg["alignment"]["method"]
    max_diff = cfg["alignment"]["max_time_diff_ms"]

    print(f"\n[INFO] cmd_vel 对齐策略: {method}, 最大时间差: {max_diff} ms")
    aligned = align_cmd_vel(image_timestamps, cmd_vel_msgs, method, max_diff)

    # 统计对齐质量
    diffs = [a["time_diff_ms"] for a in aligned if a["valid"]]
    if diffs:
        print(f"[INFO] 对齐时间差统计 (有效帧):")
        print(f"        均值: {np.mean(diffs):.2f} ms")
        print(f"        中位数: {np.median(diffs):.2f} ms")
        print(f"        最大: {np.max(diffs):.2f} ms")
        print(f"        P95: {np.percentile(diffs, 95):.2f} ms")

    # 4) 写入 labels.csv
    stats = write_labels_csv(output_dir, saved_frames, aligned, cfg)

    # 5) 写入 odom_raw.csv
    odom_count = write_odom_csv(output_dir, odom_msgs)

    # 6) 写入 meta.json
    write_meta_json(output_dir, cfg, stats, bag_path, saved_frames,
                    odom_count=odom_count)

    print(f"\n{'=' * 60}")
    print(f"  导出完成! 输出目录: {output_dir}")
    if odom_count > 0:
        print(f"  odom: {odom_count} 条 → odom_raw.csv")
    print(f"  下一步: python verify_alignment.py --data {output_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="从 ROS2 rosbag2 导出走廊导航数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python corridor_export.py --bag ./nav --output ./data/corridor_run1

  # 指定配置文件
  python corridor_export.py --bag ./nav --output ./data/corridor_run1 --config corridor_config.yaml

  # 覆盖话题名称
  python corridor_export.py --bag ./nav --output ./data/corridor_run1 \\
      --image-topic /camera/color/image_raw \\
      --cmd-vel-topic /cmd_vel

  # 使用线性插值对齐，最大允许 200ms
  python corridor_export.py --bag ./nav --output ./data/corridor_run1 \\
      --align linear_interp --max-diff 200

  # png 格式 + 每 500ms 采样
  python corridor_export.py --bag ./nav --output ./data/corridor_run1 \\
      --format png --interval 500
""")

    parser.add_argument("--bag", type=str, required=True,
                        help="ROS2 bag 路径（文件夹，含 metadata.yaml）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录路径")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径 (默认使用内置配置)")

    # 便捷覆盖参数
    parser.add_argument("--image-topic", type=str, default=None,
                        help="覆盖图像话题名称")
    parser.add_argument("--cmd-vel-topic", type=str, default=None,
                        help="覆盖 cmd_vel 话题名称")
    parser.add_argument("--odom-topic", type=str, default=None,
                        help="里程计话题名称 (默认 /odom_raw)")
    parser.add_argument("--align", type=str, choices=["nearest", "linear_interp"],
                        default=None, help="对齐方法")
    parser.add_argument("--max-diff", type=float, default=None,
                        help="最大允许时间差 (ms)")
    parser.add_argument("--format", type=str, choices=["jpg", "png"],
                        default=None, help="图像格式")
    parser.add_argument("--interval", type=int, default=None,
                        help="采样间隔 (ms)")
    parser.add_argument("--resize", type=int, nargs=2, default=None,
                        metavar=("W", "H"), help="输出图像尺寸")

    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 命令行覆盖
    if args.image_topic:
        cfg["bag"]["image_topic"] = args.image_topic
    if args.cmd_vel_topic:
        cfg["bag"]["cmd_vel_topic"] = args.cmd_vel_topic
    if args.odom_topic:
        cfg["bag"]["odom_topic"] = args.odom_topic
    if args.align:
        cfg["alignment"]["method"] = args.align
    if args.max_diff is not None:
        cfg["alignment"]["max_time_diff_ms"] = args.max_diff
    if args.format:
        cfg["image"]["format"] = args.format
    if args.interval is not None:
        cfg["image"]["sample_interval_ms"] = args.interval
    if args.resize:
        cfg["image"]["resize"] = args.resize

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 执行导出
    run_export(args.bag, args.output, cfg)


if __name__ == "__main__":
    main()
