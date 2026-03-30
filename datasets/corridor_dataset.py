"""
走廊导航数据集读取模块
================================
适配 corridor_export.py 生成的数据格式。

支持：
  - CorridorDataset: 单帧模式（离散分类 / 连续回归）
  - CorridorSequenceDataset: 序列模式（SNN 跨帧状态学习）
  - 5 类 / 3 类动作映射
  - valid-only 过滤
  - 多 run 自动聚合
  - class_weights / WeightedRandomSampler 计算

示例:
    from datasets.corridor_dataset import CorridorDataset, CorridorSequenceDataset

    # 离散 5 类
    ds = CorridorDataset('data/', mode='discrete', action_set='5')

    # 离散 3 类 (Left / Straight / Right)
    ds = CorridorDataset('data/', mode='discrete', action_set='3')

    # 回归 angular_z
    ds = CorridorDataset('data/', mode='regression', control_dim=1)

    # 序列模式 (SNN)
    sds = CorridorSequenceDataset('data/', seq_len=8, stride=1, mode='discrete')
"""

import os
import csv
import json
import warnings
from collections import Counter, OrderedDict
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image

# ============================================================================
# 常量
# ============================================================================
# 5 类动作 (与 corridor_export.py 一致)
ACTION_5CLASS = {
    0: 'Forward',
    1: 'Backward',
    2: 'Left',
    3: 'Right',
    4: 'Stop',
}

# 5→3 类映射表 (可配置 backward_policy)
# 3 类: 0=Left, 1=Straight, 2=Right
MAP_5TO3_DEFAULT = {
    0: 1,   # Forward  → Straight
    1: -1,  # Backward → 丢弃 (默认)
    2: 0,   # Left     → Left
    3: 2,   # Right    → Right
    4: 1,   # Stop     → Straight
}

ACTION_3CLASS = {
    0: 'Left',
    1: 'Straight',
    2: 'Right',
}


# ============================================================================
# 工具函数
# ============================================================================
def _find_runs(root_dir: str) -> List[str]:
    """
    自动发现 root_dir 下的所有 run 目录。

    匹配规则:
      1. root_dir 本身包含 images/ + labels.csv → 单 run
      2. root_dir 下的子目录包含 images/ + labels.csv → 多 run
    """
    runs = []

    # 检查 root_dir 本身
    if (os.path.isdir(os.path.join(root_dir, 'images')) and
            os.path.isfile(os.path.join(root_dir, 'labels.csv'))):
        runs.append(root_dir)
        return runs

    # 扫描子目录
    for entry in sorted(os.listdir(root_dir)):
        run_dir = os.path.join(root_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        if (os.path.isdir(os.path.join(run_dir, 'images')) and
                os.path.isfile(os.path.join(run_dir, 'labels.csv'))):
            runs.append(run_dir)

    return runs


def _load_labels_csv(csv_path: str) -> List[Dict[str, Any]]:
    """读取 labels.csv，返回行字典列表"""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for ridx, row in enumerate(reader):
            image_name = row.get('image_name', '')
            # frame_idx 规则：
            # 1) 优先 labels.csv 中已有字段
            # 2) 否则用 image_name 去后缀
            # 3) 再兜底到 run 内行号
            frame_idx = row.get('frame_idx', '')
            if frame_idx in (None, '') and image_name:
                frame_idx = os.path.splitext(image_name)[0]
            if frame_idx in (None, ''):
                frame_idx = str(ridx)

            # t_rel_ms 可能不存在或为空，做安全解析
            t_rel_raw = row.get('t_rel_ms', '')
            try:
                t_rel_ms = float(t_rel_raw) if t_rel_raw not in ('', None) else ''
            except (TypeError, ValueError):
                t_rel_ms = ''

            rows.append({
                'image_name': image_name,
                'action_id': int(row['action_id']),
                'action_name': row['action_name'],
                'timestamp_ns': int(row['timestamp_ns']),
                'linear_x': float(row['linear_x']),
                'angular_z': float(row['angular_z']),
                'time_diff_ms': float(row.get('time_diff_ms', 0.0)),
                'valid': int(row['valid']),
                # 以下字段为可选元信息，缺失时保持空串/默认值
                'phase': row.get('phase', ''),
                't_rel_ms': t_rel_ms,
                'run_name': row.get('run_name', ''),
                'split': row.get('split', ''),
                'frame_idx': frame_idx,
            })
    return rows


def _build_3class_map(backward_policy: str = 'drop') -> Dict[int, int]:
    """
    构建 5→3 类映射。

    backward_policy:
      'drop'    — 丢弃 Backward 帧 (映射值=-1)
      'straight' — Backward 合并到 Straight
    """
    m = dict(MAP_5TO3_DEFAULT)
    if backward_policy == 'straight':
        m[1] = 1  # Backward → Straight
    elif backward_policy == 'drop':
        m[1] = -1  # Backward → 丢弃
    else:
        raise ValueError(f"未知 backward_policy: '{backward_policy}'，"
                         f"支持: 'drop', 'straight'")
    return m


def compute_class_weights(labels: List[int],
                          num_classes: int) -> torch.Tensor:
    """
    根据标签频率计算 inverse-frequency 权重。

    Returns:
        class_weights: shape [num_classes], float32 tensor
    """
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        # inverse frequency, 避免除零
        w = total / (num_classes * max(cnt, 1))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: List[int],
                           num_classes: int) -> WeightedRandomSampler:
    """
    构建 WeightedRandomSampler，解决动作类别不平衡。

    Returns:
        sampler: 可直接传给 DataLoader(sampler=...)
    """
    cw = compute_class_weights(labels, num_classes)
    sample_weights = [cw[l].item() for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )


def print_action_distribution(labels: List[int],
                              action_names: Dict[int, str],
                              title: str = "动作分布"):
    """打印动作分布统计"""
    counts = Counter(labels)
    total = len(labels)
    print(f'\n  {title} (共 {total} 帧)')
    print('  ' + '-' * 50)
    for aid in sorted(counts.keys()):
        cnt = counts[aid]
        pct = cnt / total * 100
        bar = '█' * int(pct / 2)
        name = action_names.get(aid, f'class_{aid}')
        print(f'    {name:10s} (id={aid}): {cnt:6d}  ({pct:5.1f}%)  {bar}')
    print('  ' + '-' * 50)


# ============================================================================
# CorridorDataset — 单帧模式
# ============================================================================
class CorridorDataset(Dataset):
    """
    走廊导航数据集（单帧模式）。

    Args:
        root_dir: run 目录 或 包含多个 run 子目录的根目录
        mode: 'discrete' (分类) 或 'regression' (回归)
        control_dim: 回归时的控制维度
            1 = 只回归 angular_z
            2 = 回归 [linear_x, angular_z]
        valid_only: True=丢弃 valid=0 的帧
        action_set: '5' (5类) 或 '3' (3类: Left/Straight/Right)
        backward_policy: action_set='3' 时 Backward 的处理
            'drop' = 丢弃 Backward 帧
            'straight' = 合并到 Straight
        transforms: torchvision transforms
        return_meta: True=额外返回 meta dict
        print_stats: True=构造时打印动作分布
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = 'discrete',
        control_dim: int = 1,
        valid_only: bool = True,
        action_set: str = '5',
        backward_policy: str = 'drop',
        transforms=None,
        return_meta: bool = False,
        print_stats: bool = True,
    ):
        assert mode in ('discrete', 'regression'), \
            f"mode 必须是 'discrete' 或 'regression', 得到: '{mode}'"
        assert action_set in ('5', '3'), \
            f"action_set 必须是 '5' 或 '3', 得到: '{action_set}'"
        assert control_dim in (1, 2), \
            f"control_dim 必须是 1 或 2, 得到: {control_dim}"

        self.mode = mode
        self.control_dim = control_dim
        self.valid_only = valid_only
        self.action_set = action_set
        self.transforms = transforms
        # return_meta 仅用于评估/分析阶段，默认关闭以避免影响训练吞吐
        self.return_meta = return_meta

        # 5→3 类映射
        self.map_5to3 = _build_3class_map(backward_policy) \
            if action_set == '3' else None

        # 动作名称
        if action_set == '5':
            self.action_names = ACTION_5CLASS
            self.num_classes = 5
        else:
            self.action_names = ACTION_3CLASS
            self.num_classes = 3

        # 搜集所有 run
        runs = _find_runs(root_dir)
        if not runs:
            raise FileNotFoundError(
                f"在 '{root_dir}' 下未找到有效的 run 目录 "
                f"(需要包含 images/ 和 labels.csv)")

        # 加载并过滤所有样本
        self.samples = []  # list of dict
        self.labels = []   # list of int (discrete) 用于采样器

        skipped_invalid = 0
        skipped_missing = 0
        skipped_backward = 0

        for run_dir in runs:
            csv_path = os.path.join(run_dir, 'labels.csv')
            rows = _load_labels_csv(csv_path)
            run_name_fallback = os.path.basename(run_dir)
            split_fallback = os.path.basename(os.path.dirname(run_dir))

            for row in rows:
                # 过滤 valid=0
                if valid_only and row['valid'] == 0:
                    skipped_invalid += 1
                    continue

                # 图片路径
                img_path = os.path.join(run_dir, 'images',
                                        row['image_name'])
                if not os.path.isfile(img_path):
                    skipped_missing += 1
                    continue

                # 动作映射
                if action_set == '3':
                    mapped = self.map_5to3.get(row['action_id'], -1)
                    if mapped < 0:
                        skipped_backward += 1
                        continue
                    label = mapped
                else:
                    label = row['action_id']

                sample = {
                    'img_path': img_path,
                    'image_name': row.get('image_name', ''),
                    'label': label,
                    'linear_x': row['linear_x'],
                    'angular_z': row['angular_z'],
                    'timestamp_ns': row['timestamp_ns'],
                    'time_diff_ms': row['time_diff_ms'],
                    'phase': row.get('phase', ''),
                    't_rel_ms': row.get('t_rel_ms', ''),
                    'run_name': row.get('run_name', '') or run_name_fallback,
                    'split': row.get('split', '') or split_fallback,
                    'frame_idx': row.get('frame_idx', ''),
                    'run_dir': run_dir,
                    'action_id_orig': row['action_id'],
                }
                self.samples.append(sample)
                self.labels.append(label)

        if not self.samples:
            raise RuntimeError(
                f"数据集为空！"
                f"(跳过: invalid={skipped_invalid}, "
                f"missing={skipped_missing}, "
                f"backward={skipped_backward})")

        if print_stats:
            print(f'\n[CorridorDataset] 加载完成')
            print(f'  Runs:    {len(runs)}')
            print(f'  模式:    {mode} | 动作集: {action_set}类')
            print(f'  样本数:  {len(self.samples)}')
            print(f'  跳过:    invalid={skipped_invalid}, '
                  f'missing={skipped_missing}, '
                  f'backward={skipped_backward}')
            if mode == 'discrete':
                print_action_distribution(
                    self.labels, self.action_names,
                    f'动作分布 ({action_set}类)')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 加载图像
        img = Image.open(s['img_path']).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        # 标签
        if self.mode == 'discrete':
            label = torch.tensor(s['label'], dtype=torch.int64)
        else:
            if self.control_dim == 1:
                label = torch.tensor([s['angular_z']], dtype=torch.float32)
            else:
                label = torch.tensor([s['linear_x'], s['angular_z']],
                                     dtype=torch.float32)

        if self.return_meta:
            meta = {
                'image_name': s.get('image_name', ''),
                'timestamp_ns': s['timestamp_ns'],
                'time_diff_ms': s['time_diff_ms'],
                'linear_x': s['linear_x'],
                'angular_z': s['angular_z'],
                'phase': s.get('phase', ''),
                't_rel_ms': s.get('t_rel_ms', ''),
                'run_name': s.get('run_name', ''),
                'split': s.get('split', ''),
                'frame_idx': s.get('frame_idx', ''),
                'run_dir': s['run_dir'],
                'action_id_orig': s['action_id_orig'],
            }
            return img, label, meta
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """返回 inverse-frequency 类别权重 (用于损失函数加权)"""
        return compute_class_weights(self.labels, self.num_classes)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """返回 WeightedRandomSampler (用于 DataLoader)"""
        return build_weighted_sampler(self.labels, self.num_classes)


# ============================================================================
# CorridorSequenceDataset — 序列模式
# ============================================================================
class CorridorSequenceDataset(Dataset):
    """
    走廊导航序列数据集（SNN 跨帧状态学习）。

    从 CorridorDataset 的样本中按滑动窗口切出长度为 seq_len 的序列。

    关键：序列边界按 run 目录切分，不跨 run 拼接。
    可选按 timestamp 连续性检查，跳过时间间隔过大的断点。

    Args:
        root_dir: 同 CorridorDataset
        seq_len: 序列长度 L
        stride: 滑动窗口步长 S
        max_gap_ms: 序列内相邻帧最大允许时间间隔 (ms)
            超过此值则不创建跨越该间隔的序列。
            None = 不检查时间间隔
        mode: 'discrete' 或 'regression'
        control_dim: 同 CorridorDataset
        valid_only: 同 CorridorDataset
        action_set: 同 CorridorDataset
        backward_policy: 同 CorridorDataset
        transforms: torchvision transforms
        return_meta: True 时返回每帧 meta_list（评估/分析用）
        print_stats: True=打印统计
    """

    def __init__(
        self,
        root_dir: str,
        seq_len: int = 8,
        stride: int = 1,
        max_gap_ms: Optional[float] = 500.0,
        mode: str = 'discrete',
        control_dim: int = 1,
        valid_only: bool = True,
        action_set: str = '5',
        backward_policy: str = 'drop',
        transforms=None,
        return_meta: bool = False,
        print_stats: bool = True,
    ):
        assert seq_len >= 2, f"seq_len 必须 >= 2, 得到: {seq_len}"
        self.seq_len = seq_len
        self.stride = stride
        self.max_gap_ms = max_gap_ms
        self.mode = mode
        self.control_dim = control_dim
        self.transforms = transforms
        # return_meta 仅用于评估/分析阶段，默认关闭以避免影响训练吞吐
        self.return_meta = return_meta

        if action_set == '5':
            self.num_classes = 5
            self.action_names = ACTION_5CLASS
        else:
            self.num_classes = 3
            self.action_names = ACTION_3CLASS

        # 先用 CorridorDataset 加载所有帧
        base_ds = CorridorDataset(
            root_dir=root_dir,
            mode=mode,
            control_dim=control_dim,
            valid_only=valid_only,
            action_set=action_set,
            backward_policy=backward_policy,
            transforms=None,  # transforms 在 __getitem__ 中逐帧做
            return_meta=False,
            print_stats=False,
        )

        # 按 run_dir 分组
        run_groups = OrderedDict()
        for i, s in enumerate(base_ds.samples):
            rd = s['run_dir']
            if rd not in run_groups:
                run_groups[rd] = []
            run_groups[rd].append(i)

        # 切序列：在每个 run 内滑动窗口
        self.sequences = []  # list of list[int] (base_ds 的 indices)
        skipped_gap = 0

        for rd, indices in run_groups.items():
            # 按 timestamp 排序 (理论上 CSV 已排序，但保险起见)
            indices.sort(key=lambda i: base_ds.samples[i]['timestamp_ns'])

            n = len(indices)
            for start in range(0, n - seq_len + 1, stride):
                seq_indices = indices[start: start + seq_len]

                # 检查时间连续性
                if max_gap_ms is not None:
                    gap_ok = True
                    for j in range(len(seq_indices) - 1):
                        s_curr = base_ds.samples[seq_indices[j]]
                        s_next = base_ds.samples[seq_indices[j + 1]]
                        gap = abs(s_next['timestamp_ns'] -
                                  s_curr['timestamp_ns']) / 1e6  # ns→ms
                        if gap > max_gap_ms:
                            gap_ok = False
                            skipped_gap += 1
                            break
                    if not gap_ok:
                        continue

                self.sequences.append(seq_indices)

        self.base_samples = base_ds.samples
        self.base_labels = base_ds.labels

        if not self.sequences:
            raise RuntimeError(
                f"序列数据集为空！"
                f"(base样本={len(base_ds)}, seq_len={seq_len}, "
                f"stride={stride}, 跳过gap={skipped_gap})")

        if print_stats:
            print(f'\n[CorridorSequenceDataset] 加载完成')
            print(f'  Runs:        {len(run_groups)}')
            print(f'  base 帧数:   {len(base_ds)}')
            print(f'  序列数:      {len(self.sequences)}')
            print(f'  seq_len:     {seq_len}')
            print(f'  stride:      {stride}')
            print(f'  max_gap_ms:  {max_gap_ms}')
            print(f'  跳过(gap):   {skipped_gap}')
            if mode == 'discrete':
                print_action_distribution(
                    base_ds.labels, self.action_names,
                    f'动作分布 ({action_set}类)')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_indices = self.sequences[idx]
        frames = []
        labels = []
        metas = [] if self.return_meta else None

        for si in seq_indices:
            s = self.base_samples[si]

            img = Image.open(s['img_path']).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            elif not isinstance(img, torch.Tensor):
                # 默认转 tensor
                import torchvision.transforms.functional as TF
                img = TF.to_tensor(img)

            frames.append(img)

            if self.mode == 'discrete':
                labels.append(s['label'])
            else:
                if self.control_dim == 1:
                    labels.append([s['angular_z']])
                else:
                    labels.append([s['linear_x'], s['angular_z']])

            if self.return_meta:
                metas.append({
                    'image_name': s.get('image_name', ''),
                    'timestamp_ns': s['timestamp_ns'],
                    'linear_x': s['linear_x'],
                    'angular_z': s['angular_z'],
                    # phase 名称保持原样，不做改写
                    'phase': s.get('phase', ''),
                    't_rel_ms': s.get('t_rel_ms', ''),
                    'run_name': s.get('run_name', ''),
                    'split': s.get('split', ''),
                    'frame_idx': s.get('frame_idx', ''),
                })

        # 堆叠: frames [L, C, H, W], labels [L] or [L, control_dim]
        frames_tensor = torch.stack(frames, dim=0)

        if self.mode == 'discrete':
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

        if self.return_meta:
            return frames_tensor, labels_tensor, metas
        return frames_tensor, labels_tensor
