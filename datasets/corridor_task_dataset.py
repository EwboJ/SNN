"""
走廊导航阶段一任务数据集读取模块
================================
读取 derive_stage1_datasets.py 生成的派生数据集:
  - action3_balanced_v1 (3类: Left/Straight/Right)
  - junction_lr_v1      (2类: Left/Right)
  - stage4_v1           (4类: Follow/Approach/Turn/Recover)

派生数据集的 labels.csv 中 label_id 已经是最终标签，
本模块 **不再做** 5→3 映射等操作，直接读取 label_id 作为标签。

示例:
    from datasets.corridor_task_dataset import (
        CorridorTaskDataset, CorridorTaskSequenceDataset)

    # 单帧
    ds = CorridorTaskDataset('data/stage1/action3_balanced_v1/train',
                              transforms=train_transform)

    # 序列
    sds = CorridorTaskSequenceDataset(
        'data/stage1/stage4_v1/train',
        seq_len=8, stride=1, transforms=train_transform)
"""

import os
import csv
import json
from collections import Counter, OrderedDict
from typing import Optional, Dict, List, Any

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image


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

    if (os.path.isdir(os.path.join(root_dir, 'images')) and
            os.path.isfile(os.path.join(root_dir, 'labels.csv'))):
        runs.append(root_dir)
        return runs

    for entry in sorted(os.listdir(root_dir)):
        run_dir = os.path.join(root_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        if (os.path.isdir(os.path.join(run_dir, 'images')) and
                os.path.isfile(os.path.join(run_dir, 'labels.csv'))):
            runs.append(run_dir)

    return runs


def _load_task_labels_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    读取派生 labels.csv。

    派生字段:
      image_name, label_id, label_name, timestamp_ns,
      linear_x, angular_z, valid,
      orig_action_id, orig_action_name,
      run_name, split, t_rel_ms, phase
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'image_name': row['image_name'],
                'label_id': int(row['label_id']),
                'label_name': row.get('label_name', ''),
                'timestamp_ns': int(row['timestamp_ns']),
                'linear_x': float(row.get('linear_x', 0)),
                'angular_z': float(row.get('angular_z', 0)),
                'valid': int(row.get('valid', 1)),
                'orig_action_id': int(row.get('orig_action_id', -1)),
                'orig_action_name': row.get('orig_action_name', ''),
                'run_name': row.get('run_name', ''),
                'split': row.get('split', ''),
                't_rel_ms': float(row.get('t_rel_ms', 0)),
                'phase': row.get('phase', ''),
            })
    return rows


def _detect_label_names(samples: list) -> Dict[int, str]:
    """从样本中自动检测 label_id → label_name 映射"""
    mapping = {}
    for s in samples:
        lid = s['label_id']
        lname = s['label_name']
        if lid not in mapping and lname:
            mapping[lid] = lname
    return dict(sorted(mapping.items()))


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
        w = total / (num_classes * max(cnt, 1))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: List[int],
                           num_classes: int) -> WeightedRandomSampler:
    """构建 WeightedRandomSampler"""
    cw = compute_class_weights(labels, num_classes)
    sample_weights = [cw[l].item() for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


def print_label_distribution(labels: List[int],
                             label_names: Dict[int, str],
                             title: str = "标签分布"):
    """打印标签分布统计"""
    counts = Counter(labels)
    total = len(labels)
    print(f'\n  {title} (共 {total} 帧)')
    print('  ' + '-' * 50)
    for lid in sorted(counts.keys()):
        cnt = counts[lid]
        pct = cnt / total * 100
        bar = '█' * int(pct / 2)
        name = label_names.get(lid, f'class_{lid}')
        print(f'    {name:10s} (id={lid}): {cnt:6d}  ({pct:5.1f}%)  {bar}')
    print('  ' + '-' * 50)


# ============================================================================
# CorridorTaskDataset — 单帧模式
# ============================================================================

class CorridorTaskDataset(Dataset):
    """
    阶段一派生数据集（单帧模式）。

    直接从 labels.csv 中读取 label_id 作为最终标签，
    不做任何额外映射。

    Args:
        root_dir: split 目录 (如 data/stage1/action3_balanced_v1/train/)
                  或包含 images/ + labels.csv 的单 run 目录
        transforms: torchvision transforms
        return_meta: True 时 __getitem__ 返回 (img, label, meta)
        print_stats: True 时构造后打印标签分布
    """

    def __init__(
        self,
        root_dir: str,
        transforms=None,
        return_meta: bool = False,
        print_stats: bool = True,
    ):
        self.root_dir = root_dir
        self.transforms = transforms
        self.return_meta = return_meta

        runs = _find_runs(root_dir)
        if not runs:
            raise FileNotFoundError(
                f"在 '{root_dir}' 下未找到有效的 run 目录 "
                f"(需要包含 images/ 和 labels.csv)")

        self.samples = []
        self.labels = []
        skipped_missing = 0

        for run_dir in runs:
            csv_path = os.path.join(run_dir, 'labels.csv')
            rows = _load_task_labels_csv(csv_path)

            for row in rows:
                img_path = os.path.join(run_dir, 'images',
                                        row['image_name'])
                if not os.path.isfile(img_path):
                    skipped_missing += 1
                    continue

                sample = {
                    'img_path': img_path,
                    'label_id': row['label_id'],
                    'label_name': row['label_name'],
                    'timestamp_ns': row['timestamp_ns'],
                    'linear_x': row['linear_x'],
                    'angular_z': row['angular_z'],
                    'orig_action_id': row['orig_action_id'],
                    'orig_action_name': row['orig_action_name'],
                    'run_name': row['run_name'],
                    'split': row['split'],
                    't_rel_ms': row['t_rel_ms'],
                    'phase': row['phase'],
                    'run_dir': run_dir,
                }
                self.samples.append(sample)
                self.labels.append(row['label_id'])

        if not self.samples:
            raise RuntimeError(
                f"数据集为空！(missing={skipped_missing})")

        # 自动检测类别
        self.label_names = _detect_label_names(self.samples)
        self.num_classes = max(self.labels) + 1

        if print_stats:
            print(f'\n[CorridorTaskDataset] 加载完成')
            print(f'  Runs:    {len(runs)}')
            print(f'  样本数:  {len(self.samples)}')
            print(f'  类别数:  {self.num_classes}')
            if skipped_missing > 0:
                print(f'  跳过:    missing={skipped_missing}')
            print_label_distribution(
                self.labels, self.label_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = Image.open(s['img_path']).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = torch.tensor(s['label_id'], dtype=torch.int64)

        if self.return_meta:
            meta = {
                'timestamp_ns': s['timestamp_ns'],
                'linear_x': s['linear_x'],
                'angular_z': s['angular_z'],
                'run_name': s['run_name'],
                'run_dir': s['run_dir'],
                'orig_action_id': s['orig_action_id'],
                'orig_action_name': s['orig_action_name'],
                'label_name': s['label_name'],
                'phase': s['phase'],
                't_rel_ms': s['t_rel_ms'],
            }
            return img, label, meta
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """返回 inverse-frequency 类别权重"""
        return compute_class_weights(self.labels, self.num_classes)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """返回 WeightedRandomSampler"""
        return build_weighted_sampler(self.labels, self.num_classes)


# ============================================================================
# CorridorTaskSequenceDataset — 序列模式
# ============================================================================

class CorridorTaskSequenceDataset(Dataset):
    """
    阶段一派生序列数据集（SNN 跨帧状态学习）。

    在同一 run 内按滑动窗口切序列，不跨 run 拼接。

    Args:
        root_dir: split 目录
        seq_len: 序列长度 L
        stride: 滑动窗口步长
        max_gap_ms: 相邻帧最大时间间隔 (ms), None=不检查
        transforms: torchvision transforms
        return_meta: True 时 __getitem__ 返回 (frames, labels, meta_list)
        print_stats: True 时打印统计
    """

    def __init__(
        self,
        root_dir: str,
        seq_len: int = 8,
        stride: int = 1,
        max_gap_ms: Optional[float] = 500.0,
        transforms=None,
        return_meta: bool = False,
        print_stats: bool = True,
    ):
        assert seq_len >= 2, f"seq_len 必须 >= 2, 得到: {seq_len}"
        self.seq_len = seq_len
        self.stride = stride
        self.max_gap_ms = max_gap_ms
        self.transforms = transforms
        self.return_meta = return_meta

        # 加载全部帧
        base_ds = CorridorTaskDataset(
            root_dir=root_dir,
            transforms=None,
            return_meta=False,
            print_stats=False,
        )

        self.num_classes = base_ds.num_classes
        self.label_names = base_ds.label_names

        # 按 run_dir 分组
        run_groups = OrderedDict()
        for i, s in enumerate(base_ds.samples):
            rd = s['run_dir']
            if rd not in run_groups:
                run_groups[rd] = []
            run_groups[rd].append(i)

        # 滑动窗口切序列
        self.sequences = []
        skipped_gap = 0

        for rd, indices in run_groups.items():
            indices.sort(key=lambda i: base_ds.samples[i]['timestamp_ns'])
            n = len(indices)

            for start in range(0, n - seq_len + 1, stride):
                seq_idx = indices[start: start + seq_len]

                if max_gap_ms is not None:
                    gap_ok = True
                    for j in range(len(seq_idx) - 1):
                        s_c = base_ds.samples[seq_idx[j]]
                        s_n = base_ds.samples[seq_idx[j + 1]]
                        gap = abs(s_n['timestamp_ns'] -
                                  s_c['timestamp_ns']) / 1e6
                        if gap > max_gap_ms:
                            gap_ok = False
                            skipped_gap += 1
                            break
                    if not gap_ok:
                        continue

                self.sequences.append(seq_idx)

        self.base_samples = base_ds.samples
        self.base_labels = base_ds.labels

        if not self.sequences:
            raise RuntimeError(
                f"序列数据集为空！"
                f"(base样本={len(base_ds)}, seq_len={seq_len}, "
                f"stride={stride}, 跳过gap={skipped_gap})")

        if print_stats:
            print(f'\n[CorridorTaskSequenceDataset] 加载完成')
            print(f'  Runs:        {len(run_groups)}')
            print(f'  base 帧数:   {len(base_ds)}')
            print(f'  序列数:      {len(self.sequences)}')
            print(f'  seq_len:     {seq_len}')
            print(f'  stride:      {stride}')
            print(f'  max_gap_ms:  {max_gap_ms}')
            print(f'  跳过(gap):   {skipped_gap}')
            print_label_distribution(
                base_ds.labels, self.label_names)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_idx = self.sequences[idx]
        frames = []
        labels = []
        metas = [] if self.return_meta else None

        for si in seq_idx:
            s = self.base_samples[si]

            img = Image.open(s['img_path']).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            elif not isinstance(img, torch.Tensor):
                import torchvision.transforms.functional as TF
                img = TF.to_tensor(img)

            frames.append(img)
            labels.append(s['label_id'])

            if self.return_meta:
                metas.append({
                    'timestamp_ns': s['timestamp_ns'],
                    'label_name': s['label_name'],
                    'phase': s['phase'],
                    't_rel_ms': s['t_rel_ms'],
                    'run_name': s['run_name'],
                })

        frames_tensor = torch.stack(frames, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        if self.return_meta:
            return frames_tensor, labels_tensor, metas
        return frames_tensor, labels_tensor

    def get_class_weights(self) -> torch.Tensor:
        """返回 inverse-frequency 类别权重"""
        return compute_class_weights(self.base_labels, self.num_classes)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """返回 WeightedRandomSampler (基于所有序列的首帧标签)"""
        first_labels = [self.base_samples[seq[0]]['label_id']
                        for seq in self.sequences]
        return build_weighted_sampler(first_labels, self.num_classes)
