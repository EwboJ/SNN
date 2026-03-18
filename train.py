"""
统一训练入口 - SNN 论文消融实验框架 (v3)
=========================================
支持:
  - CIFAR-10/100（原流程完全不变）
  - 走廊导航原始数据集（corridor, 离散/回归/序列）
  - 阶段一派生任务（corridor_task, action3/junction_lr/stage4）

v3 变更:
  - 新增 --dataset corridor_task 分支
  - 新增 --task_root / --task_name / --task_num_classes 参数
  - 新增 --label_smoothing / --focal_gamma 参数
  - 支持 CorridorTaskDataset / CorridorTaskSequenceDataset

CIFAR-10:
    python train.py -T 8 --neuron_type APLIF --residual_mode ADD --seed 42 -epochs 151

走廊导航原始 (3类):
    python train.py --dataset corridor --corridor_root ./data/corridor \\
        --mode discrete --action_set 3 --encoding rate -T 4 \\
        --class_balance weighted_sampler -b 32 -epochs 80

阶段一派生任务 (action3_balanced):
    python train.py --dataset corridor_task \\
        --task_root ./data/stage1/action3_balanced_v1 \\
        --task_name action3_balanced --task_num_classes 3 \\
        --encoding rate -T 4 --neuron_type APLIF --residual_mode ADD \\
        --class_balance weighted_sampler -b 32 -epochs 80 \\
        -enable_tensorboard --final_test

阶段一派生任务 (junction_lr):
    python train.py --dataset corridor_task \\
        --task_root ./data/stage1/junction_lr_v1 \\
        --task_name junction_lr --task_num_classes 2 \\
        --encoding rate -T 8 -b 32 -epochs 60 --final_test

阶段一派生任务 (stage4):
    python train.py --dataset corridor_task \\
        --task_root ./data/stage1/stage4_v1 \\
        --task_name stage4 --task_num_classes 4 \\
        --encoding rate -T 4 -b 32 -epochs 80 --final_test
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import torchvision
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import BaseNode
from torch.utils.data import DataLoader
import torch.utils.data as data
import time
import json
import argparse
import math
from tqdm import tqdm
import numpy as np
from ADD_ResNet110 import resnet110
from torch.utils.tensorboard import SummaryWriter

# seed 将在 main() 中根据命令行参数设置


# ============================================================================
# Spike 统计 Monitor（可反传版本）
# ============================================================================
class SpikeMonitor:
    """
    收集所有 BaseNode 子类的脉冲输出，支持：
    - 可反传的 spike_rate 正则（保留计算图的 tensor）
    - 分层发放率统计（用于日志的 .item() 版本）
    - sparsity / total spike count 统计
    """
    def __init__(self):
        self.spike_tensors = {}   # name -> spike tensor（保留计算图）
        self.spike_counts = {}    # name -> (total_spikes, total_elements)
        self.handles = []

    def register(self, net: nn.Module):
        """对网络中所有 BaseNode 子类注册 hook（兼容 LIF/ALIF/PLIF/APLIF）"""
        for name, module in net.named_modules():
            if isinstance(module, BaseNode):
                handle = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # 保留计算图，用于可反传正则化
                self.spike_tensors[name] = output
                # 用 detach 版本做日志统计
                with torch.no_grad():
                    self.spike_counts[name] = (
                        output.sum().item(),
                        output.numel()
                    )
        return hook

    def reset(self):
        self.spike_tensors.clear()
        self.spike_counts.clear()

    def get_avg_spike_rate_tensor(self):
        """返回 tensor 形式的平均发放率（保留梯度，可反传）"""
        if not self.spike_tensors:
            return None
        rates = [t.mean() for t in self.spike_tensors.values()]
        return torch.stack(rates).mean()

    def get_avg_spike_rate(self):
        """返回 float 形式的平均发放率（仅统计/日志用）"""
        if not self.spike_counts:
            return 0.0
        total_s = sum(v[0] for v in self.spike_counts.values())
        total_e = sum(v[1] for v in self.spike_counts.values())
        return total_s / total_e if total_e > 0 else 0.0

    def get_sparsity(self):
        """返回全网 sparsity = 1 - avg_spike_rate"""
        return 1.0 - self.get_avg_spike_rate()

    def get_layer_rates(self):
        """返回分层发放率 dict: {layer_prefix: rate}"""
        layer_stats = {}
        for name, (spikes, elements) in self.spike_counts.items():
            parts = name.split('.')
            layer_key = parts[0] if parts[0].startswith('layer') else 'stem'
            if layer_key not in layer_stats:
                layer_stats[layer_key] = [0.0, 0]
            layer_stats[layer_key][0] += spikes
            layer_stats[layer_key][1] += elements
        return {k: v[0] / v[1] if v[1] > 0 else 0.0
                for k, v in layer_stats.items()}

    def get_total_spike_count(self):
        """返回总 spike 数量（能耗 proxy）"""
        return sum(v[0] for v in self.spike_counts.values())

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ============================================================================
# 数据集工厂
# ============================================================================
def build_cifar_dataset(dataset_name, data_dir):
    """构建 CIFAR-10/100 数据集（与原逻辑完全一致）"""
    if dataset_name == 'cifar10':
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
        ])
        train_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, transform=test_transform, download=True)
        return train_ds, test_ds, 10, 3

    elif dataset_name == 'cifar100':
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
        train_ds = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, transform=test_transform, download=True)
        return train_ds, test_ds, 100, 3
    else:
        raise ValueError(f"未知 CIFAR 数据集: {dataset_name}")


def build_corridor_dataset(args):
    """
    构建走廊导航数据集 (train / val / test 三划分)。

    Returns:
        train_ds, val_ds, test_ds, num_classes_or_dim, in_channels,
        is_sequence, sampler_or_None
    """
    from datasets.corridor_dataset import (
        CorridorDataset, CorridorSequenceDataset)

    img_h = args.img_h
    img_w = args.img_w

    # 训练增强
    augments = [
        torchvision.transforms.Resize((img_h, img_w)),
    ]
    # ⚠ corridor_hflip: 对 Left/Right 标签有语义影响！
    # 如果 mode=discrete 且 action_set=3，水平翻转会把 Left 场景变成 Right 场景，
    # 但标签不会自动互换。如需启用，必须在 dataset 层同步做 Left/Right 标签互换。
    # 默认关闭(false)以避免标签不一致。
    if args.corridor_hflip:
        augments.append(torchvision.transforms.RandomHorizontalFlip())
        print("  ⚠ corridor_hflip=True: 已启用水平翻转增强")
        if args.mode == 'discrete' and args.action_set == '3':
            print("    警告: discrete+3cls 下水平翻转需要 dataset 层同步互换 "
                  "Left/Right 标签，否则标签不一致!")
    augments.extend([
        torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    train_transform = torchvision.transforms.Compose(augments)

    # val/test 无增强
    eval_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_h, img_w)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    mode = args.mode
    action_set = args.action_set
    control_dim = args.control_dim
    valid_only = args.valid_only
    corridor_root = args.corridor_root

    # 确定 num_classes
    if mode == 'discrete':
        num_out = 3 if action_set == '3' else 5
    else:
        num_out = control_dim

    is_sequence = args.seq_len > 0
    sampler = None

    # 路径: <corridor_root>/train, <corridor_root>/val, <corridor_root>/test
    train_root = os.path.join(corridor_root, 'train')
    val_root = os.path.join(corridor_root, 'val')
    test_root = os.path.join(corridor_root, 'test')

    if not os.path.isdir(train_root):
        raise FileNotFoundError(
            f"训练目录不存在: {train_root}\n"
            f"请先运行 scripts/split_corridor_runs.py 划分数据")

    has_val = os.path.isdir(val_root)
    has_test = os.path.isdir(test_root)

    if not has_val:
        print("  ⚠ 未找到 val/ 目录，将使用 test/ 作为 val")
        val_root = test_root

    if is_sequence:
        train_ds = CorridorSequenceDataset(
            root_dir=train_root,
            seq_len=args.seq_len, stride=args.stride,
            mode=mode, control_dim=control_dim,
            valid_only=valid_only, action_set=action_set,
            transforms=train_transform)
        val_ds = CorridorSequenceDataset(
            root_dir=val_root,
            seq_len=args.seq_len, stride=args.stride,
            mode=mode, control_dim=control_dim,
            valid_only=valid_only, action_set=action_set,
            transforms=eval_transform)
        test_ds = CorridorSequenceDataset(
            root_dir=test_root,
            seq_len=args.seq_len, stride=args.stride,
            mode=mode, control_dim=control_dim,
            valid_only=valid_only, action_set=action_set,
            transforms=eval_transform) if has_test else None
    else:
        train_ds = CorridorDataset(
            root_dir=train_root,
            mode=mode, control_dim=control_dim,
            valid_only=valid_only, action_set=action_set,
            transforms=train_transform)
        val_ds = CorridorDataset(
            root_dir=val_root,
            mode=mode, control_dim=control_dim,
            valid_only=valid_only, action_set=action_set,
            transforms=eval_transform)
        test_ds = CorridorDataset(
            root_dir=test_root,
            mode=mode, control_dim=control_dim,
            valid_only=valid_only, action_set=action_set,
            transforms=eval_transform) if has_test else None

        # 类别平衡 (仅 train, 仅 discrete 单帧)
        if mode == 'discrete' and args.class_balance == 'weighted_sampler':
            sampler = train_ds.get_weighted_sampler()

    return train_ds, val_ds, test_ds, num_out, 3, is_sequence, sampler


def build_corridor_task_dataset(args):
    """
    构建阶段一派生任务数据集 (train / val / test)。
    labels.csv 中 label_id 直接作为最终标签。

    Returns:
        train_ds, val_ds, test_ds, num_classes, in_channels,
        is_sequence, sampler_or_None
    """
    from datasets.corridor_task_dataset import (
        CorridorTaskDataset, CorridorTaskSequenceDataset)

    img_h = args.img_h
    img_w = args.img_w

    # 训练增强
    augments = [
        torchvision.transforms.Resize((img_h, img_w)),
    ]
    if args.corridor_hflip:
        augments.append(torchvision.transforms.RandomHorizontalFlip())
        print("  ⚠ corridor_hflip=True: 已启用水平翻转增强")
    augments.extend([
        torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    train_transform = torchvision.transforms.Compose(augments)

    eval_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_h, img_w)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    task_root = args.task_root
    num_classes = args.task_num_classes
    is_sequence = args.seq_len > 0
    sampler = None

    train_root = os.path.join(task_root, 'train')
    val_root = os.path.join(task_root, 'val')
    test_root = os.path.join(task_root, 'test')

    if not os.path.isdir(train_root):
        raise FileNotFoundError(
            f"训练目录不存在: {train_root}\n"
            f"请先运行 scripts/derive_stage1_datasets.py 派生数据")

    has_val = os.path.isdir(val_root)
    has_test = os.path.isdir(test_root)
    if not has_val:
        print("  ⚠ 未找到 val/ 目录，将使用 test/ 作为 val")
        val_root = test_root

    if is_sequence:
        train_ds = CorridorTaskSequenceDataset(
            root_dir=train_root,
            seq_len=args.seq_len, stride=args.stride,
            transforms=train_transform)
        val_ds = CorridorTaskSequenceDataset(
            root_dir=val_root,
            seq_len=args.seq_len, stride=args.stride,
            transforms=eval_transform)
        test_ds = CorridorTaskSequenceDataset(
            root_dir=test_root,
            seq_len=args.seq_len, stride=args.stride,
            transforms=eval_transform) if has_test else None
    else:
        train_ds = CorridorTaskDataset(
            root_dir=train_root, transforms=train_transform)
        val_ds = CorridorTaskDataset(
            root_dir=val_root, transforms=eval_transform)
        test_ds = CorridorTaskDataset(
            root_dir=test_root, transforms=eval_transform) if has_test else None

        # 类别平衡
        if args.class_balance == 'weighted_sampler':
            sampler = train_ds.get_weighted_sampler()

    return train_ds, val_ds, test_ds, num_classes, 3, is_sequence, sampler


# ============================================================================
# Regression 专用: final_test 增强评估
# ============================================================================

def _run_regression_final_test(net, loader, is_sequence, args):
    """
    Regression 模式下的增强 final_test 评估。

    在常规 MAE 之外, 额外统计:
      - RMSE (全局)
      - 若 DataLoader 的 Dataset 支持 return_meta 且含 phase 字段:
        - Correcting 阶段 MAE / RMSE
        - Settled    阶段 MAE / RMSE

    不改变训练主循环, 仅在 final_test 阶段被调用。

    Returns:
        dict: {
            test_rmse, phase_available,
            phase_stats: {Correcting: {mae, rmse, count},
                          Settled:    {mae, rmse, count}}
        }
    """
    import math
    from collections import defaultdict

    net.eval()

    # ---------- 检查 dataset 是否支持 return_meta ----------
    ds = loader.dataset
    has_meta = hasattr(ds, 'return_meta')
    old_return_meta = None
    if has_meta:
        old_return_meta = ds.return_meta
        ds.return_meta = True    # 临时开启

    # ---------- 收集所有预测与标签 ----------
    all_errors = []              # (pred - label) 逐样本
    phase_errors = defaultdict(list)   # phase -> list of abs_error
    phase_sq_errors = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            if has_meta and len(batch) == 3:
                frame, label, meta = batch
            else:
                frame, label = batch[:2]
                meta = None

            frame = frame.float().to(args.device)
            label = label.to(args.device)

            if is_sequence:
                B, L = frame.shape[0], frame.shape[1]
                if hasattr(net, 'reset_state'):
                    net.reset_state()
                for t in range(L):
                    out_t = net(frame[:, t])
                    err = (out_t - label[:, t].float()).cpu()
                    for i in range(B):
                        e = err[i].item()
                        all_errors.append(e)
                        if meta is not None:
                            # meta 是 list[dict] 或 list[list[dict]]
                            ph = _extract_phase(meta, i, t)
                            if ph:
                                phase_errors[ph].append(abs(e))
                                phase_sq_errors[ph].append(e * e)
                    if args.encoding == 'rate' and hasattr(net, 'backbone'):
                        functional.reset_net(net.backbone)
            else:
                out_fr = net(frame)
                err = (out_fr - label.float()).cpu()
                B = err.shape[0]
                for i in range(B):
                    e = err[i].item()
                    all_errors.append(e)
                    if meta is not None:
                        ph = _extract_phase(meta, i)
                        if ph:
                            phase_errors[ph].append(abs(e))
                            phase_sq_errors[ph].append(e * e)
                functional.reset_net(net)

    # ---------- 恢复 return_meta ----------
    if has_meta and old_return_meta is not None:
        ds.return_meta = old_return_meta

    # ---------- 全局 RMSE ----------
    n = len(all_errors)
    mse = sum(e * e for e in all_errors) / max(n, 1)
    rmse = math.sqrt(mse)

    # ---------- Phase 统计 ----------
    phase_available = len(phase_errors) > 0
    phase_stats = {}
    for ph in ('Correcting', 'Settled'):
        if ph in phase_errors:
            cnt = len(phase_errors[ph])
            mae = sum(phase_errors[ph]) / max(cnt, 1)
            p_mse = sum(phase_sq_errors[ph]) / max(cnt, 1)
            p_rmse = math.sqrt(p_mse)
            phase_stats[ph] = {
                'mae': round(mae, 6),
                'rmse': round(p_rmse, 6),
                'count': cnt,
            }
    # 其余未知 phase
    for ph in phase_errors:
        if ph not in phase_stats:
            cnt = len(phase_errors[ph])
            mae = sum(phase_errors[ph]) / max(cnt, 1)
            p_mse = sum(phase_sq_errors[ph]) / max(cnt, 1)
            p_rmse = math.sqrt(p_mse)
            phase_stats[ph] = {
                'mae': round(mae, 6),
                'rmse': round(p_rmse, 6),
                'count': cnt,
            }

    return {
        'test_rmse': round(rmse, 6),
        'phase_available': phase_available,
        'phase_stats': phase_stats if phase_available else {},
    }


def _extract_phase(meta, sample_idx, seq_t=None):
    """
    从 meta 中提取 phase 字符串。

    meta 格式取决于 Dataset:
      - 单帧: list[dict], meta[sample_idx]['phase']
      - 序列: list[list[dict]], meta[sample_idx][seq_t]['phase']
      - 或者 dict of lists (DataLoader collate)
    """
    try:
        if isinstance(meta, dict):
            # DataLoader default_collate: dict of lists
            phases = meta.get('phase', None)
            if phases is not None:
                ph = phases[sample_idx]
                return str(ph) if ph else ''
        elif isinstance(meta, (list, tuple)):
            item = meta[sample_idx]
            if isinstance(item, dict):
                return str(item.get('phase', ''))
            elif isinstance(item, (list, tuple)) and seq_t is not None:
                return str(item[seq_t].get('phase', ''))
    except (IndexError, KeyError, TypeError):
        pass
    return ''


# ============================================================================
# 评估 helper (val / test 共用)
# ============================================================================
def run_evaluation(net, loader, compute_loss, is_discrete, is_sequence,
                   spike_monitor, args, desc='Val'):
    """
    在给定 DataLoader 上运行评估。

    Returns:
        dict: {loss, metric, avg_sr, avg_sp, avg_spk_img, ...}
    """
    net.eval()
    eval_loss = 0
    eval_acc = 0
    eval_mae = 0
    eval_samples = 0
    eval_spike_rates = []
    eval_sparsities = []
    eval_total_spikes = 0

    pbar = tqdm(loader, desc=desc)
    with torch.no_grad():
        for frame, label in pbar:
            spike_monitor.reset()
            frame = frame.float().to(args.device)
            label = label.to(args.device)

            if is_sequence:
                B, L = frame.shape[0], frame.shape[1]
                if hasattr(net, 'reset_state'):
                    net.reset_state()

                seq_loss = 0
                for t in range(L):
                    out_t = net(frame[:, t])
                    seq_loss += compute_loss(out_t, label[:, t]).item()

                    if is_discrete:
                        eval_acc += (out_t.argmax(1) == label[:, t]) \
                            .float().sum().item()
                    else:
                        eval_mae += (out_t - label[:, t].float()) \
                            .abs().sum().item()

                    if args.encoding == 'rate' and hasattr(net, 'backbone'):
                        functional.reset_net(net.backbone)

                batch_samples = B * L
                eval_loss += seq_loss / L * batch_samples
                eval_samples += batch_samples

            else:
                out_fr = net(frame)
                loss = compute_loss(out_fr, label)

                batch_samples = label.numel() if label.dim() <= 1 \
                    else label.shape[0]
                eval_samples += batch_samples
                eval_loss += loss.item() * batch_samples

                if is_discrete:
                    eval_acc += (out_fr.argmax(1) == label) \
                        .float().sum().item()
                else:
                    eval_mae += (out_fr - label.float()).abs().sum().item()

                functional.reset_net(net)

            sr = spike_monitor.get_avg_spike_rate()
            sp = spike_monitor.get_sparsity()
            eval_spike_rates.append(sr)
            eval_sparsities.append(sp)
            eval_total_spikes += spike_monitor.get_total_spike_count()

            if is_discrete:
                pbar.set_postfix({
                    'acc': f'{eval_acc/max(eval_samples,1):.4f}',
                    'sr': f'{sr:.4f}'
                })
            else:
                pbar.set_postfix({
                    'mae': f'{eval_mae/max(eval_samples,1):.4f}',
                    'sr': f'{sr:.4f}'
                })

    eval_loss /= max(eval_samples, 1)
    eval_metric = eval_acc / max(eval_samples, 1) if is_discrete \
        else eval_mae / max(eval_samples, 1)
    avg_sr = np.mean(eval_spike_rates) if eval_spike_rates else 0
    avg_sp = np.mean(eval_sparsities) if eval_sparsities else 0
    avg_spk_img = eval_total_spikes / max(eval_samples, 1)

    return {
        'loss': eval_loss,
        'metric': eval_metric,
        'avg_sr': avg_sr,
        'avg_sp': avg_sp,
        'avg_spk_img': avg_spk_img,
        'samples': eval_samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description='SNN 可迁移研究框架 - 统一训练入口',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ======================== 框架核心参数 ========================
    parser.add_argument('-T', default=8, type=int,
                        help='仿真时间步数 (simulating time-steps)')
    parser.add_argument('--neuron_type', default='APLIF', type=str,
                        choices=['LIF', 'PLIF', 'ALIF', 'APLIF'],
                        help='神经元类型')
    parser.add_argument('--residual_mode', default='ADD', type=str,
                        choices=['standard', 'ADD'],
                        help='残差连接模式')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'corridor',
                                 'corridor_task'],
                        help='数据集')
    parser.add_argument('--num_classes', default=None, type=int,
                        help='分类数 (不指定则自动确定)')
    parser.add_argument('--in_channels', default=None, type=int,
                        help='输入通道数 (不指定则自动确定)')
    parser.add_argument('--task_type', default='classification', type=str,
                        choices=['classification', 'regression'],
                        help='任务类型 (CIFAR 专用)')
    parser.add_argument('--seed', default=42, type=int,
                        help='随机种子')
    parser.add_argument('--weights_only_export', action='store_true',
                        help='训练结束后导出仅权重文件')

    # ======================== 走廊导航参数 ========================
    parser.add_argument('--corridor_root', default='./data/corridor',
                        type=str, help='走廊数据根目录 (含 train/ val/ test/)')
    parser.add_argument('--mode', default='discrete', type=str,
                        choices=['discrete', 'regression'],
                        help='走廊任务模式')
    parser.add_argument('--action_set', default='3', type=str,
                        choices=['5', '3'],
                        help='动作集: 5=全部, 3=Left/Straight/Right')
    parser.add_argument('--control_dim', default=1, type=int,
                        choices=[1, 2],
                        help='回归维度: 1=angular_z, 2=[v,w]')
    parser.add_argument('--encoding', default='rate', type=str,
                        choices=['rate', 'framediff'],
                        help='输入编码: rate=单帧重复T, framediff=帧差')
    parser.add_argument('--seq_len', default=0, type=int,
                        help='序列长度 (0=单帧训练, >0=序列训练)')
    parser.add_argument('--stride', default=1, type=int,
                        help='序列滑动窗口步长')
    parser.add_argument('--valid_only', default=True, type=lambda x:
                        str(x).lower() in ('true', '1', 'yes'),
                        help='是否丢弃 valid=0 帧')
    parser.add_argument('--loss_type', default='auto', type=str,
                        choices=['auto', 'ce', 'mse_onehot', 'huber'],
                        help='损失函数 (auto=根据任务自动选)')
    parser.add_argument('--class_balance', default='none', type=str,
                        choices=['none', 'weighted_sampler', 'class_weight'],
                        help='类别平衡策略')
    parser.add_argument('--smooth_lambda', default=0.0, type=float,
                        help='回归动作平滑正则系数')
    parser.add_argument('--spike_lambda', default=0.001, type=float,
                        help='spike rate 正则系数')
    parser.add_argument('-target_rate', default=0.1, type=float,
                        help='目标平均脉冲发放率')
    parser.add_argument('--v_max', default=0.22, type=float,
                        help='最大线速度 (m/s)')
    parser.add_argument('--w_max', default=2.84, type=float,
                        help='最大角速度 (rad/s)')

    # ======================== 走廊图像参数 ========================
    parser.add_argument('--img_h', default=32, type=int,
                        help='走廊图像输入高度 (默认32, 可改64等)')
    parser.add_argument('--img_w', default=32, type=int,
                        help='走廊图像输入宽度 (默认32, 可改96等)')
    parser.add_argument('--corridor_hflip', action='store_true',
                        help='走廊训练启用 RandomHorizontalFlip '
                             '(默认关闭, 因 Left/Right 标签不会自动互换)')

    # ======================== 阶段一派生任务参数 ========================
    parser.add_argument('--task_root', default='./data/stage1/action3_balanced_v1',
                        type=str,
                        help='corridor_task 数据根目录 (含 train/val/test/)')
    parser.add_argument('--task_name', default='action3_balanced', type=str,
                        help='任务名 (用于实验命名: action3_balanced/junction_lr/stage4)')
    parser.add_argument('--task_num_classes', default=3, type=int,
                        help='corridor_task 类别数')
    parser.add_argument('--label_smoothing', default=0.0, type=float,
                        help='CrossEntropyLoss label smoothing')
    parser.add_argument('--focal_gamma', default=0.0, type=float,
                        help='Focal Loss gamma (0.0=禁用, >0=启用)')

    # ======================== 训练参数 ========================
    parser.add_argument('-device', default='cuda:0', help='训练设备')
    parser.add_argument('-b', default=8, type=int, help='batch size')
    parser.add_argument('-epochs', default=151, type=int, metavar='N',
                        help='总训练轮数')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='数据加载线程数')
    parser.add_argument('-channels', default=128, type=int,
                        help='SNN Conv2d 通道数')
    parser.add_argument('-data_dir', type=str, default='./data/CIFAR-10',
                        help='数据集根目录 (CIFAR)')
    parser.add_argument('-out_dir', type=str, default='./checkpoint/CIFAR-10',
                        help='输出目录')
    parser.add_argument('-resume', type=str, help='从 checkpoint 恢复训练')
    parser.add_argument('-amp', action='store_true',
                        help='自动混合精度训练')
    parser.add_argument('-opt', type=str, default='Adam',
                        help='优化器 (SGD / Adam)')
    parser.add_argument('-lr', default=0.002, type=float, help='学习率')
    parser.add_argument('-momentum', default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('-lr_scheduler', default='StepLR', type=str,
                        help='学习率调度器 (StepLR / CosALR)')
    parser.add_argument('-step_size', default=40, type=float,
                        help='StepLR step_size')
    parser.add_argument('-gamma', default=0.1, type=float,
                        help='StepLR gamma')
    parser.add_argument('-T_max', default=32, type=int,
                        help='CosineAnnealingLR T_max')
    parser.add_argument('-enable_tensorboard', action='store_true',
                        help='启用 TensorBoard 日志')
    parser.add_argument('-log_interval', default=50, type=int,
                        help='日志记录间隔 (batch)')
    parser.add_argument('--final_test', action='store_true',
                        help='训练结束后对 test 集做最终评估')

    args = parser.parse_args()

    # 判断数据集类型
    is_corridor = (args.dataset == 'corridor')
    is_corridor_task = (args.dataset == 'corridor_task')

    # ======================== 设置随机种子 ========================
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ======================== 生成实验标识 ========================
    if is_corridor_task:
        seq_tag = f"_seq{args.seq_len}" if args.seq_len > 0 else ""
        exp_name = (f"corridor_task_{args.task_name}_"
                    f"{args.neuron_type}_{args.residual_mode}_T{args.T}{seq_tag}")
        out_base = os.path.join(args.out_dir, '..', 'corridor_task')
    elif is_corridor:
        task_tag = f"{args.mode}_{args.action_set}cls" if args.mode == 'discrete' \
            else f"{args.mode}_dim{args.control_dim}"
        enc_tag = args.encoding
        seq_tag = f"_seq{args.seq_len}" if args.seq_len > 0 else ""
        exp_name = (f"corridor_{task_tag}_{enc_tag}_"
                    f"{args.neuron_type}_{args.residual_mode}_T{args.T}{seq_tag}")
        out_base = os.path.join(args.out_dir, '..', 'corridor')
    else:
        exp_name = f"{args.neuron_type}_{args.residual_mode}_T{args.T}"
        out_base = args.out_dir
    exp_out_dir = os.path.join(out_base, exp_name)
    os.makedirs(exp_out_dir, exist_ok=True)

    # ======================== 打印配置 ========================
    print('=' * 70)
    print(f'  SNN 训练  [{exp_name}]')
    print('=' * 70)
    print(f'  神经元类型:     {args.neuron_type}')
    print(f'  残差模式:       {args.residual_mode}')
    print(f'  时间步 T:       {args.T}')
    print(f'  Seed:           {args.seed}')
    print(f'  数据集:         {args.dataset}')
    if is_corridor_task:
        print(f'  任务名:         {args.task_name}')
        print(f'  类别数:         {args.task_num_classes}')
        print(f'  数据根目录:     {args.task_root}')
        print(f'  图像尺寸:       {args.img_h}×{args.img_w}')
        print(f'  编码:           {args.encoding}')
        if args.label_smoothing > 0:
            print(f'  Label Smoothing: {args.label_smoothing}')
        if args.focal_gamma > 0:
            print(f'  Focal Gamma:    {args.focal_gamma}')
        print(f'  序列长度:       {args.seq_len}' if args.seq_len > 0
              else f'  训练模式:       单帧')
    elif is_corridor:
        print(f'  走廊模式:       {args.mode}')
        print(f'  编码:           {args.encoding}')
        print(f'  动作集:         {args.action_set}类' if args.mode == 'discrete'
              else f'  控制维度:       {args.control_dim}')
        print(f'  图像尺寸:       {args.img_h}×{args.img_w}')
        print(f'  水平翻转:       {"是" if args.corridor_hflip else "否 (默认)"}')
        print(f'  序列长度:       {args.seq_len}' if args.seq_len > 0
              else f'  训练模式:       单帧')
    else:
        print(f'  任务类型:       {args.task_type}')
    print(f'  设备:           {args.device}')
    print(f'  Batch Size:     {args.b}')
    print(f'  Epochs:         {args.epochs}')
    print(f'  学习率:         {args.lr}')
    print(f'  输出目录:       {exp_out_dir}')
    print('=' * 70)

    # ======================== CUDA 初始化 ========================
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.cuda.synchronize()
            print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
            print(f"初始显存: {torch.cuda.memory_allocated(0)/1024**2:.0f}MB / "
                  f"{torch.cuda.get_device_properties(0).total_mem/1024**2:.0f}MB")
        except (RuntimeError, AttributeError):
            pass

    # ======================== 数据集 ========================
    is_sequence = False
    sampler = None
    val_dataset = None
    test_dataset = None

    if is_corridor_task:
        (train_dataset, val_dataset, test_dataset, num_out,
         auto_in_channels, is_sequence, sampler) = build_corridor_task_dataset(args)
        num_classes = num_out
    elif is_corridor:
        (train_dataset, val_dataset, test_dataset, num_out,
         auto_in_channels, is_sequence, sampler) = build_corridor_dataset(args)
        num_classes = num_out
    else:
        train_dataset, test_dataset, num_classes, auto_in_channels = \
            build_cifar_dataset(args.dataset, args.data_dir)
        # CIFAR: 用 test_dataset 作 val (保持原行为兼容)
        val_dataset = test_dataset

    in_channels = args.in_channels if args.in_channels is not None \
        else auto_in_channels
    if args.num_classes is not None:
        num_classes = args.num_classes

    print(f'  输出维度:       {num_classes}')
    print(f'  输入通道:       {in_channels}')
    print(f'  训练样本数:     {len(train_dataset)}')
    print(f'  验证样本数:     {len(val_dataset)}')
    if test_dataset is not None and test_dataset is not val_dataset:
        print(f'  测试样本数:     {len(test_dataset)}')
    if is_sequence:
        print(f'  序列模式:       seq_len={args.seq_len}, stride={args.stride}')
    print('=' * 70)

    # ======================== TensorBoard ========================
    writer = None
    if args.enable_tensorboard:
        writer = SummaryWriter(os.path.join(exp_out_dir, 'runs'))

    # ======================== 构建网络 ========================
    if is_corridor_task:
        from models.snn_corridor import build_corridor_net
        net = build_corridor_net(
            head_type='discrete',
            num_actions=num_classes,
            control_dim=1,
            encoding=args.encoding,
            T=args.T,
            neuron_type=args.neuron_type,
            residual_mode=args.residual_mode,
            raw_in_channels=in_channels,
            use_tanh=False,
        )
    elif is_corridor:
        from models.snn_corridor import build_corridor_net
        net = build_corridor_net(
            head_type=args.mode,
            num_actions=(3 if args.action_set == '3' else 5),
            control_dim=args.control_dim,
            encoding=args.encoding,
            T=args.T,
            neuron_type=args.neuron_type,
            residual_mode=args.residual_mode,
            raw_in_channels=in_channels,
            use_tanh=(args.mode == 'regression'),
            v_max=args.v_max,
            w_max=args.w_max,
        )
    else:
        net = resnet110(
            num_classes=num_classes,
            T=args.T,
            neuron_type=args.neuron_type,
            residual_mode=args.residual_mode,
            in_channels=in_channels,
        )
    print(net)
    net.to(args.device)

    # ======================== Spike Monitor ========================
    spike_monitor = SpikeMonitor()
    spike_monitor.register(net)

    # ======================== 优化器 ========================
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.T_max)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(args.step_size), gamma=args.gamma)

    # ======================== 恢复训练 ========================
    start_epoch = 1
    max_val_acc = 0.0
    min_val_mae = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 从checkpoint恢复: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu',
                                    weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                net.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                max_val_acc = checkpoint.get('max_test_acc',
                              checkpoint.get('max_val_acc', 0.0))
                min_val_mae = checkpoint.get('min_test_mae',
                              checkpoint.get('min_val_mae', float('inf')))
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'lr_scheduler_state_dict' in checkpoint:
                    lr_scheduler.load_state_dict(
                        checkpoint['lr_scheduler_state_dict'])
                print(f"=> 从epoch {checkpoint.get('epoch', 0)}恢复")
            else:
                net.load_state_dict(checkpoint)

    # ======================== DataLoader ========================
    # Train: weighted_sampler 仅作用于 train
    if sampler is not None:
        train_data_loader = data.DataLoader(
            dataset=train_dataset, batch_size=args.b,
            sampler=sampler, drop_last=True, num_workers=args.j)
    else:
        train_data_loader = data.DataLoader(
            dataset=train_dataset, batch_size=args.b,
            shuffle=True, drop_last=True, num_workers=args.j)

    # Val: 始终 shuffle=False
    val_data_loader = data.DataLoader(
        dataset=val_dataset, batch_size=args.b,
        shuffle=False, drop_last=False, num_workers=args.j)

    # Test: 仅在 final_test 时使用, 始终 shuffle=False
    test_data_loader = None
    if test_dataset is not None and test_dataset is not val_dataset:
        test_data_loader = data.DataLoader(
            dataset=test_dataset, batch_size=args.b,
            shuffle=False, drop_last=False, num_workers=args.j)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    # ======================== 损失函数 ========================
    loss_type = args.loss_type
    if loss_type == 'auto':
        if is_corridor_task:
            loss_type = 'ce'
        elif is_corridor:
            loss_type = 'ce' if args.mode == 'discrete' else 'huber'
        else:
            loss_type = 'mse_onehot'

    # CE class weights
    ce_weight = None
    if is_corridor_task and args.class_balance == 'class_weight' \
            and not is_sequence:
        ce_weight = train_dataset.get_class_weights().to(args.device)
        print(f'  类别权重: {ce_weight.tolist()}')
    elif is_corridor and args.mode == 'discrete' and \
            args.class_balance == 'class_weight' and not is_sequence:
        ce_weight = train_dataset.get_class_weights().to(args.device)
        print(f'  类别权重: {ce_weight.tolist()}')

    # CIFAR class weights (与原始设置一致)
    cifar_class_weights = None
    if not is_corridor and not is_corridor_task and args.dataset == 'cifar10':
        cifar_class_weights = torch.FloatTensor([
            1.0, 1.0, 1.2, 2.0, 1.1, 1.5, 1.0, 1.0, 1.0, 1.0
        ]).to(args.device)

    ce_criterion = None
    huber_criterion = None
    if loss_type == 'ce':
        ce_criterion = nn.CrossEntropyLoss(
            weight=ce_weight,
            label_smoothing=args.label_smoothing)
    elif loss_type == 'huber':
        huber_criterion = nn.SmoothL1Loss(beta=0.1)

    def compute_loss(out, label):
        """统一损失计算"""
        if loss_type == 'ce':
            return ce_criterion(out, label)
        elif loss_type == 'huber':
            return huber_criterion(out, label.float())
        else:  # mse_onehot (CIFAR 原始路径)
            label_onehot = F.one_hot(label, num_classes).float()
            if cifar_class_weights is not None:
                mse = F.mse_loss(out, label_onehot, reduction='none')
                return (mse * cifar_class_weights.unsqueeze(0)).mean()
            return F.mse_loss(out, label_onehot)

    is_discrete = is_corridor_task or \
                  (is_corridor and args.mode == 'discrete') or \
                  (not is_corridor and not is_corridor_task and
                   args.task_type == 'classification')

    # ======================== config 快照 ========================
    os.makedirs(exp_out_dir, exist_ok=True)
    config = {
        'neuron_type': args.neuron_type,
        'residual_mode': args.residual_mode,
        'T': args.T,
        'num_classes': num_classes,
        'in_channels': in_channels,
        'exp_name': exp_name,
        'seed': args.seed,
        'dataset': args.dataset,
    }
    if is_corridor_task:
        config.update({
            'task_root': args.task_root,
            'task_name': args.task_name,
            'task_num_classes': args.task_num_classes,
            'encoding': args.encoding,
            'seq_len': args.seq_len,
            'img_h': args.img_h,
            'img_w': args.img_w,
            'label_smoothing': args.label_smoothing,
            'focal_gamma': args.focal_gamma,
        })
    elif is_corridor:
        config.update({
            'mode': args.mode,
            'encoding': args.encoding,
            'action_set': args.action_set,
            'control_dim': args.control_dim,
            'seq_len': args.seq_len,
            'img_h': args.img_h,
            'img_w': args.img_w,
            'corridor_hflip': args.corridor_hflip,
        })

    # ======================== 训练循环 ========================
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()

        start_time = time.time()
        net.train()
        train_loss = 0
        train_cls_loss = 0
        train_reg_loss_total = 0
        train_smooth_loss_total = 0
        train_acc = 0
        train_mae = 0
        train_samples = 0
        epoch_spike_rates = []
        epoch_sparsities = []

        train_pbar = tqdm(train_data_loader,
                          desc=f'Epoch {epoch}/{args.epochs-1} [Train]')
        for batch_idx, (frame, label) in enumerate(train_pbar):
            spike_monitor.reset()
            optimizer.zero_grad()

            frame = frame.float().to(args.device)
            label = label.to(args.device)

            # ---- 序列模式 ----
            if is_sequence:
                B, L = frame.shape[0], frame.shape[1]
                if hasattr(net, 'reset_state'):
                    net.reset_state()

                seq_loss = torch.tensor(0.0, device=args.device)
                seq_acc = 0
                seq_mae = 0
                prev_out = None

                for t in range(L):
                    frame_t = frame[:, t]
                    label_t = label[:, t]
                    out_t = net(frame_t)
                    step_loss = compute_loss(out_t, label_t)
                    seq_loss = seq_loss + step_loss

                    if args.smooth_lambda > 0 and prev_out is not None \
                            and not is_discrete:
                        smooth_loss = args.smooth_lambda * \
                            F.mse_loss(out_t, prev_out)
                        seq_loss = seq_loss + smooth_loss

                    if is_discrete:
                        seq_acc += (out_t.argmax(1) == label_t).float().sum().item()
                    else:
                        seq_mae += (out_t - label_t.float()).abs().sum().item()

                    prev_out = out_t.detach()

                    if args.encoding == 'rate' and hasattr(net, 'backbone'):
                        functional.reset_net(net.backbone)

                cls_loss = seq_loss / L
                batch_samples = B * L

                sr_tensor = spike_monitor.get_avg_spike_rate_tensor()
                if sr_tensor is not None and args.spike_lambda > 0:
                    reg_loss = args.spike_lambda * \
                        (sr_tensor - args.target_rate).square()
                else:
                    reg_loss = torch.tensor(0.0, device=args.device)

                loss = cls_loss + reg_loss
                loss.backward()
                optimizer.step()

                train_samples += batch_samples
                train_loss += loss.item() * batch_samples
                train_cls_loss += cls_loss.item() * batch_samples
                train_reg_loss_total += reg_loss.item() * batch_samples

                if is_discrete:
                    train_acc += seq_acc
                else:
                    train_mae += seq_mae

                functional.reset_net(net)

            # ---- 单帧模式 ----
            else:
                out_fr = net(frame)
                cls_loss = compute_loss(out_fr, label)

                sr_tensor = spike_monitor.get_avg_spike_rate_tensor()
                if sr_tensor is not None and args.spike_lambda > 0:
                    reg_loss = args.spike_lambda * \
                        (sr_tensor - args.target_rate).square()
                else:
                    reg_loss = torch.tensor(0.0, device=args.device)

                loss = cls_loss + reg_loss
                loss.backward()
                optimizer.step()

                batch_samples = label.numel() if label.dim() <= 1 \
                    else label.shape[0]
                train_samples += batch_samples
                train_loss += loss.item() * batch_samples
                train_cls_loss += cls_loss.item() * batch_samples
                train_reg_loss_total += reg_loss.item() * batch_samples

                if is_discrete:
                    if loss_type == 'ce':
                        train_acc += (out_fr.argmax(1) == label).float().sum().item()
                    else:
                        train_acc += (out_fr.argmax(1) == label).float().sum().item()
                else:
                    train_mae += (out_fr - label.float()).abs().sum().item()

                functional.reset_net(net)

            # 统计
            avg_sr = spike_monitor.get_avg_spike_rate()
            sparsity = spike_monitor.get_sparsity()
            epoch_spike_rates.append(avg_sr)
            epoch_sparsities.append(sparsity)

            if is_discrete:
                metric_str = f'acc={train_acc/max(train_samples,1):.4f}'
            else:
                metric_str = f'mae={train_mae/max(train_samples,1):.4f}'

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'metric': metric_str,
                'sr': f'{avg_sr:.4f}',
                'sp': f'{sparsity:.2%}'
            })

            if writer and batch_idx % args.log_interval == 0:
                gs = epoch * len(train_data_loader) + batch_idx
                writer.add_scalar('Train/BatchLoss', loss.item(), gs)
                writer.add_scalar('Train/BatchSpikeRate', avg_sr, gs)
                if is_discrete:
                    writer.add_scalar('Train/BatchAcc',
                        train_acc/max(train_samples,1), gs)
                else:
                    writer.add_scalar('Train/BatchMAE',
                        train_mae/max(train_samples,1), gs)

            spike_monitor.reset()

        train_loss /= max(train_samples, 1)
        train_cls_loss /= max(train_samples, 1)
        train_reg_loss_total /= max(train_samples, 1)
        train_metric = train_acc / max(train_samples, 1) if is_discrete \
            else train_mae / max(train_samples, 1)
        avg_train_sr = np.mean(epoch_spike_rates) if epoch_spike_rates else 0
        avg_train_sp = np.mean(epoch_sparsities) if epoch_sparsities else 0

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ======================== 验证 (Val) ========================
        val_result = run_evaluation(
            net, val_data_loader, compute_loss, is_discrete, is_sequence,
            spike_monitor, args,
            desc=f'Epoch {epoch}/{args.epochs-1} [Val]')

        val_loss = val_result['loss']
        val_metric = val_result['metric']
        avg_val_sr = val_result['avg_sr']
        avg_val_sp = val_result['avg_sp']
        avg_spk_img = val_result['avg_spk_img']
        epoch_time = time.time() - start_time

        # TensorBoard
        if writer:
            writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            writer.add_scalar('Epoch/TrainClsLoss', train_cls_loss, epoch)
            writer.add_scalar('Epoch/TrainRegLoss', train_reg_loss_total, epoch)
            writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            writer.add_scalar('Epoch/TrainSpikeRate', avg_train_sr, epoch)
            writer.add_scalar('Epoch/ValSpikeRate', avg_val_sr, epoch)
            writer.add_scalar('Epoch/TrainSparsity', avg_train_sp, epoch)
            writer.add_scalar('Epoch/ValSparsity', avg_val_sp, epoch)
            writer.add_scalar('Epoch/SpikesPerImage', avg_spk_img, epoch)
            writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
            writer.add_scalar('Epoch/Time', epoch_time, epoch)
            if is_discrete:
                writer.add_scalar('Epoch/TrainAcc', train_metric, epoch)
                writer.add_scalar('Epoch/ValAcc', val_metric, epoch)
            else:
                writer.add_scalar('Epoch/TrainMAE', train_metric, epoch)
                writer.add_scalar('Epoch/ValMAE', val_metric, epoch)

        # ======================== 保存模型 (按 val) ========================
        save_max = False
        if is_discrete and val_metric > max_val_acc:
            max_val_acc = val_metric
            save_max = True
        elif not is_discrete and val_metric < min_val_mae:
            min_val_mae = val_metric
            save_max = True

        if save_max:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'max_val_acc': max_val_acc,
                'min_val_mae': min_val_mae,
                'config': config,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(exp_out_dir, 'best_model.ckpt'))
            torch.save(net.state_dict(),
                       os.path.join(exp_out_dir, 'best_weights.pth'))
            best_str = f'acc={val_metric:.4f}' if is_discrete \
                else f'mae={val_metric:.4f}'
            print(f'>>> Saved BEST [{exp_name}] epoch={epoch} val_{best_str}')

        if epoch % 10 == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'max_val_acc': max_val_acc,
                'min_val_mae': min_val_mae,
                'config': config,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(
                exp_out_dir, f'checkpoint_epoch_{epoch}.ckpt'))

        # ======================== 输出训练信息 ========================
        metric_name = 'Acc' if is_discrete else 'MAE'
        best_val = max_val_acc if is_discrete else min_val_mae
        print('=' * 80)
        print(f'Epoch: {epoch}/{args.epochs-1}  [{exp_name}]')
        print(f'Train - Loss: {train_loss:.4f} '
              f'(Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss_total:.6f}), '
              f'{metric_name}: {train_metric:.4f}')
        print(f'        SpikeRate: {avg_train_sr:.4f}, '
              f'Sparsity: {avg_train_sp:.2%}')
        print(f'Val   - Loss: {val_loss:.4f}, '
              f'{metric_name}: {val_metric:.4f}')
        print(f'        SpikeRate: {avg_val_sr:.4f}, '
              f'Sparsity: {avg_val_sp:.2%}, '
              f'Spikes/Img: {avg_spk_img:.0f}')
        print(f'Best Val {metric_name}: {best_val:.4f}, '
              f'LR: {current_lr:.6f}, Time: {epoch_time:.2f}s')
        print('=' * 80)

    # ======================== 保存最终模型 ========================
    final_ckpt = {
        'epoch': args.epochs - 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'max_val_acc': max_val_acc,
        'min_val_mae': min_val_mae,
        'config': config,
        'args': vars(args),
    }
    torch.save(final_ckpt, os.path.join(exp_out_dir, 'final_model.ckpt'))
    print(f'>>> Saved final model [{exp_name}]')

    if args.weights_only_export:
        wpath = os.path.join(exp_out_dir, 'best_weights.pth')
        if not os.path.exists(wpath):
            torch.save(net.state_dict(), wpath)
            print(f'>>> Exported weights-only: {wpath}')

    # ======================== 最终测试 (可选) ========================
    if args.final_test and test_data_loader is not None:
        print('\n' + '=' * 70)
        print(f'  最终测试 [{exp_name}]')
        print('=' * 70)

        # 加载 best_model
        best_path = os.path.join(exp_out_dir, 'best_model.ckpt')
        if os.path.isfile(best_path):
            best_ckpt = torch.load(best_path, map_location='cpu',
                                   weights_only=False)
            net.load_state_dict(best_ckpt['model_state_dict'])
            print(f'  已加载 best_model (epoch={best_ckpt.get("epoch", "?")})')
        else:
            print(f'  ⚠ 未找到 best_model.ckpt，使用最终模型')

        net.to(args.device)
        test_result = run_evaluation(
            net, test_data_loader, compute_loss, is_discrete, is_sequence,
            spike_monitor, args,
            desc=f'Final Test [{exp_name}]')

        metric_name = 'Acc' if is_discrete else 'MAE'
        print(f'\n  Test {metric_name}: {test_result["metric"]:.4f}')
        print(f'  Test Loss:     {test_result["loss"]:.4f}')
        print(f'  Spike Rate:    {test_result["avg_sr"]:.4f}')
        print(f'  Sparsity:      {test_result["avg_sp"]:.2%}')
        print(f'  Spikes/Img:    {test_result["avg_spk_img"]:.0f}')
        print(f'  Samples:       {test_result["samples"]}')

        # ---- Regression 增强: RMSE + Phase 统计 ----
        reg_extra = {}
        if not is_discrete:
            print(f'\n  [Regression 增强评估]')
            reg_result = _run_regression_final_test(
                net, test_data_loader, is_sequence, args)
            reg_extra['test_rmse'] = reg_result['test_rmse']
            print(f'  Test RMSE:     {reg_result["test_rmse"]:.4f}')

            if reg_result['phase_available']:
                reg_extra['phase_available'] = True
                reg_extra['phase_stats'] = reg_result['phase_stats']
                print(f'  Phase 统计:')
                for ph, st in reg_result['phase_stats'].items():
                    print(f'    {ph:12s}  '
                          f'MAE={st["mae"]:.4f}  '
                          f'RMSE={st["rmse"]:.4f}  '
                          f'n={st["count"]}')
            else:
                reg_extra['phase_available'] = False
                print(f'  Phase 信息: 不可用 (dataset 无 phase 元数据)')

        # 保存 final_test_metrics.json
        test_metrics = {
            'exp_name': exp_name,
            f'test_{metric_name.lower()}': round(test_result['metric'], 6),
            'test_loss': round(test_result['loss'], 6),
            'test_spike_rate': round(test_result['avg_sr'], 6),
            'test_sparsity': round(test_result['avg_sp'], 6),
            'test_spikes_per_image': round(test_result['avg_spk_img'], 1),
            'test_samples': test_result['samples'],
            'best_epoch': best_ckpt.get('epoch', None) if os.path.isfile(best_path) else None,
            f'best_val_{metric_name.lower()}': round(
                max_val_acc if is_discrete else min_val_mae, 6),
            'config': config,
        }
        # 合并 regression 增强字段
        test_metrics.update(reg_extra)

        test_json_path = os.path.join(exp_out_dir, 'final_test_metrics.json')
        with open(test_json_path, 'w', encoding='utf-8') as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)
        print(f'  [✓] {test_json_path}')
        print('=' * 70)

    if writer:
        writer.close()

    best_val = max_val_acc if is_discrete else min_val_mae
    metric_name = 'Acc' if is_discrete else 'MAE'
    print(f'\n训练完成! [{exp_name}] 最佳 Val {metric_name}: {best_val:.4f}')


if __name__ == '__main__':
    main()
