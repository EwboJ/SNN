"""
统一训练入口 - SNN 论文消融实验框架
====================================
支持 CIFAR-10/100（原流程完全不变）+ 走廊导航数据集（离散/回归/序列）。

CIFAR-10 消融示例 (与原来完全一致):
    python train.py -T 8 --neuron_type APLIF --residual_mode ADD --seed 42 -epochs 151
    python train.py --neuron_type LIF --residual_mode ADD -T 8 --seed 42

走廊导航 离散分类 (3类: Left/Straight/Right):
    python train.py --dataset corridor --corridor_root ./data/corridor \\
        --mode discrete --action_set 3 --encoding rate -T 8

走廊导航 连续回归 (angular_z):
    python train.py --dataset corridor --corridor_root ./data/corridor \\
        --mode regression --control_dim 1 --encoding framediff -T 1

走廊导航 序列训练 (SNN 跨帧状态学习):
    python train.py --dataset corridor --corridor_root ./data/corridor \\
        --mode discrete --seq_len 8 --stride 2
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
    构建走廊导航数据集。

    Returns:
        train_ds, test_ds, num_classes_or_dim, in_channels, is_sequence, sampler_or_None
    """
    from datasets.corridor_dataset import (
        CorridorDataset, CorridorSequenceDataset)

    img_size = 32  # 匹配 ResNet 输入

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    mode = args.mode  # 'discrete' or 'regression'
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

    if is_sequence:
        train_ds = CorridorSequenceDataset(
            root_dir=os.path.join(corridor_root, 'train'),
            seq_len=args.seq_len, stride=args.stride,
            mode=mode, control_dim=control_dim,
            valid_only=valid_only, action_set=action_set,
            transforms=train_transform)
        test_ds = CorridorSequenceDataset(
            root_dir=os.path.join(corridor_root, 'test'),
            seq_len=args.seq_len, stride=args.stride,
            mode=mode, control_dim=control_dim,
            valid_only=valid_only, action_set=action_set,
            transforms=test_transform)
    else:
        train_ds = CorridorDataset(
            root_dir=os.path.join(corridor_root, 'train'),
            mode=mode, control_dim=control_dim,
            valid_only=valid_only, action_set=action_set,
            transforms=train_transform)
        test_ds = CorridorDataset(
            root_dir=os.path.join(corridor_root, 'test'),
            mode=mode, control_dim=control_dim,
            valid_only=valid_only, action_set=action_set,
            transforms=test_transform)

        # 类别平衡 (仅 discrete 单帧)
        if mode == 'discrete' and args.class_balance == 'weighted_sampler':
            sampler = train_ds.get_weighted_sampler()

    return train_ds, test_ds, num_out, 3, is_sequence, sampler


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
                        choices=['cifar10', 'cifar100', 'corridor'],
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
                        type=str, help='走廊数据根目录')
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

    args = parser.parse_args()

    # 判断是否走廊训练
    is_corridor = (args.dataset == 'corridor')

    # ======================== 设置随机种子 ========================
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ======================== 生成实验标识 ========================
    if is_corridor:
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
    if is_corridor:
        print(f'  走廊模式:       {args.mode}')
        print(f'  编码:           {args.encoding}')
        print(f'  动作集:         {args.action_set}类' if args.mode == 'discrete'
              else f'  控制维度:       {args.control_dim}')
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
                  f"{torch.cuda.get_device_properties(0).total_memory/1024**2:.0f}MB")
        except RuntimeError as e:
            print(f"CUDA初始化错误: {e}")
            exit(1)

    # ======================== 数据集 ========================
    is_sequence = False
    sampler = None

    if is_corridor:
        (train_dataset, test_dataset, num_out, auto_in_channels,
         is_sequence, sampler) = build_corridor_dataset(args)
        num_classes = num_out  # discrete: num_actions; regression: control_dim
    else:
        train_dataset, test_dataset, num_classes, auto_in_channels = \
            build_cifar_dataset(args.dataset, args.data_dir)

    in_channels = args.in_channels if args.in_channels is not None \
        else auto_in_channels
    if args.num_classes is not None:
        num_classes = args.num_classes

    print(f'  输出维度:       {num_classes}')
    print(f'  输入通道:       {in_channels}')
    print(f'  训练样本数:     {len(train_dataset)}')
    print(f'  测试样本数:     {len(test_dataset)}')
    if is_sequence:
        print(f'  序列模式:       seq_len={args.seq_len}, stride={args.stride}')
    print('=' * 70)

    # ======================== TensorBoard ========================
    writer = None
    if args.enable_tensorboard:
        writer = SummaryWriter(os.path.join(exp_out_dir, 'runs'))

    # ======================== 构建网络 ========================
    if is_corridor:
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
    max_test_acc = 0.0
    min_test_mae = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 从checkpoint恢复: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu',
                                    weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                net.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                max_test_acc = checkpoint.get('max_test_acc', 0.0)
                min_test_mae = checkpoint.get('min_test_mae', float('inf'))
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'lr_scheduler_state_dict' in checkpoint:
                    lr_scheduler.load_state_dict(
                        checkpoint['lr_scheduler_state_dict'])
                print(f"=> 从epoch {checkpoint.get('epoch', 0)}恢复")
            else:
                net.load_state_dict(checkpoint)

    # ======================== DataLoader ========================
    if sampler is not None:
        train_data_loader = data.DataLoader(
            dataset=train_dataset, batch_size=args.b,
            sampler=sampler, drop_last=True, num_workers=args.j)
    else:
        train_data_loader = data.DataLoader(
            dataset=train_dataset, batch_size=args.b,
            shuffle=True, drop_last=True, num_workers=args.j)
    test_data_loader = data.DataLoader(
        dataset=test_dataset, batch_size=args.b,
        shuffle=False, drop_last=False, num_workers=args.j)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    # ======================== 损失函数 ========================
    # 决定 loss_type
    loss_type = args.loss_type
    if loss_type == 'auto':
        if is_corridor:
            loss_type = 'ce' if args.mode == 'discrete' else 'huber'
        else:
            loss_type = 'mse_onehot'

    # CE class weights (corridor discrete + class_weight)
    ce_weight = None
    if is_corridor and args.mode == 'discrete' and \
            args.class_balance == 'class_weight' and not is_sequence:
        ce_weight = train_dataset.get_class_weights().to(args.device)
        print(f'  类别权重: {ce_weight.tolist()}')

    # CIFAR class weights (与原始设置一致)
    cifar_class_weights = None
    if not is_corridor and args.dataset == 'cifar10':
        cifar_class_weights = torch.FloatTensor([
            1.0, 1.0, 1.2, 2.0, 1.1, 1.5, 1.0, 1.0, 1.0, 1.0
        ]).to(args.device)

    ce_criterion = None
    huber_criterion = None
    if loss_type == 'ce':
        ce_criterion = nn.CrossEntropyLoss(weight=ce_weight)
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

    is_discrete = (is_corridor and args.mode == 'discrete') or \
                  (not is_corridor and args.task_type == 'classification')

    # ======================== 确保输出目录存在 ========================
    os.makedirs(exp_out_dir, exist_ok=True)
    # ======================== config 快照 (训练前确定，避免 UnboundLocalError) ========================
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
    if is_corridor:
        config.update({
            'mode': args.mode,
            'encoding': args.encoding,
            'action_set': args.action_set,
            'control_dim': args.control_dim,
            'seq_len': args.seq_len,
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
                # frame: [B, L, C, H, W], label: [B, L] or [B, L, D]
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

                    if hasattr(net, 'forward'):
                        out_t = net(frame_t)
                    else:
                        out_t = net(frame_t)

                    step_loss = compute_loss(out_t, label_t)
                    seq_loss = seq_loss + step_loss

                    # 动作平滑正则 (回归模式)
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

                    # rate 模式序列内：每步 reset backbone
                    if args.encoding == 'rate' and hasattr(net, 'backbone'):
                        functional.reset_net(net.backbone)
                    # framediff：不 reset，SNN 状态跨帧累积

                cls_loss = seq_loss / L
                batch_samples = B * L

                # Spike 正则 (在序列最后一步的状态上)
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

            # ---- 单帧模式 ----
            else:
                out_fr = net(frame)
                cls_loss = compute_loss(out_fr, label)

                # Spike 正则 (可反传)
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

        # ======================== 测试 ========================
        net.eval()
        test_loss = 0
        test_acc = 0
        test_mae = 0
        test_samples = 0
        test_spike_rates = []
        test_sparsities = []
        test_total_spikes = 0

        test_pbar = tqdm(test_data_loader,
                         desc=f'Epoch {epoch}/{args.epochs-1} [Test]')
        with torch.no_grad():
            for frame, label in test_pbar:
                spike_monitor.reset()
                frame = frame.float().to(args.device)
                label = label.to(args.device)

                if is_sequence:
                    # 序列测试
                    B, L = frame.shape[0], frame.shape[1]
                    if hasattr(net, 'reset_state'):
                        net.reset_state()

                    seq_loss = 0
                    for t in range(L):
                        out_t = net(frame[:, t])
                        seq_loss += compute_loss(out_t, label[:, t]).item()

                        if is_discrete:
                            test_acc += (out_t.argmax(1) == label[:, t]) \
                                .float().sum().item()
                        else:
                            test_mae += (out_t - label[:, t].float()) \
                                .abs().sum().item()

                        if args.encoding == 'rate' and hasattr(net, 'backbone'):
                            functional.reset_net(net.backbone)

                    batch_samples = B * L
                    test_loss += seq_loss / L * batch_samples
                    test_samples += batch_samples

                else:
                    # 单帧测试
                    out_fr = net(frame)
                    loss = compute_loss(out_fr, label)

                    batch_samples = label.numel() if label.dim() <= 1 \
                        else label.shape[0]
                    test_samples += batch_samples
                    test_loss += loss.item() * batch_samples

                    if is_discrete:
                        if loss_type == 'ce':
                            test_acc += (out_fr.argmax(1) == label) \
                                .float().sum().item()
                        else:
                            test_acc += (out_fr.argmax(1) == label) \
                                .float().sum().item()
                    else:
                        test_mae += (out_fr - label.float()).abs().sum().item()

                    functional.reset_net(net)

                sr = spike_monitor.get_avg_spike_rate()
                sp = spike_monitor.get_sparsity()
                test_spike_rates.append(sr)
                test_sparsities.append(sp)
                test_total_spikes += spike_monitor.get_total_spike_count()

                if is_discrete:
                    test_pbar.set_postfix({
                        'acc': f'{test_acc/max(test_samples,1):.4f}',
                        'sr': f'{sr:.4f}'
                    })
                else:
                    test_pbar.set_postfix({
                        'mae': f'{test_mae/max(test_samples,1):.4f}',
                        'sr': f'{sr:.4f}'
                    })

        test_loss /= max(test_samples, 1)
        test_metric = test_acc / max(test_samples, 1) if is_discrete \
            else test_mae / max(test_samples, 1)
        avg_test_sr = np.mean(test_spike_rates) if test_spike_rates else 0
        avg_test_sp = np.mean(test_sparsities) if test_sparsities else 0
        avg_spk_img = test_total_spikes / max(test_samples, 1)
        epoch_time = time.time() - start_time

        # TensorBoard
        if writer:
            writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            writer.add_scalar('Epoch/TrainClsLoss', train_cls_loss, epoch)
            writer.add_scalar('Epoch/TrainRegLoss', train_reg_loss_total, epoch)
            writer.add_scalar('Epoch/TestLoss', test_loss, epoch)
            writer.add_scalar('Epoch/TrainSpikeRate', avg_train_sr, epoch)
            writer.add_scalar('Epoch/TestSpikeRate', avg_test_sr, epoch)
            writer.add_scalar('Epoch/TrainSparsity', avg_train_sp, epoch)
            writer.add_scalar('Epoch/TestSparsity', avg_test_sp, epoch)
            writer.add_scalar('Epoch/SpikesPerImage', avg_spk_img, epoch)
            writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
            writer.add_scalar('Epoch/Time', epoch_time, epoch)
            if is_discrete:
                writer.add_scalar('Epoch/TrainAcc', train_metric, epoch)
                writer.add_scalar('Epoch/TestAcc', test_metric, epoch)
            else:
                writer.add_scalar('Epoch/TrainMAE', train_metric, epoch)
                writer.add_scalar('Epoch/TestMAE', test_metric, epoch)

        # ======================== 保存模型 ========================
        save_max = False
        if is_discrete and test_metric > max_test_acc:
            max_test_acc = test_metric
            save_max = True
        elif not is_discrete and test_metric < min_test_mae:
            min_test_mae = test_metric
            save_max = True

        if save_max:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'max_test_acc': max_test_acc,
                'min_test_mae': min_test_mae,
                'config': config,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(exp_out_dir, 'best_model.ckpt'))
            torch.save(net.state_dict(),
                       os.path.join(exp_out_dir, 'best_weights.pth'))
            best_str = f'acc={test_metric:.4f}' if is_discrete \
                else f'mae={test_metric:.4f}'
            print(f'>>> Saved BEST [{exp_name}] epoch={epoch} {best_str}')

        if epoch % 10 == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'max_test_acc': max_test_acc,
                'min_test_mae': min_test_mae,
                'config': config,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(
                exp_out_dir, f'checkpoint_epoch_{epoch}.ckpt'))

        # ======================== 输出训练信息 ========================
        metric_name = 'Acc' if is_discrete else 'MAE'
        best_val = max_test_acc if is_discrete else min_test_mae
        print('=' * 80)
        print(f'Epoch: {epoch}/{args.epochs-1}  [{exp_name}]')
        print(f'Train - Loss: {train_loss:.4f} '
              f'(Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss_total:.6f}), '
              f'{metric_name}: {train_metric:.4f}')
        print(f'        SpikeRate: {avg_train_sr:.4f}, '
              f'Sparsity: {avg_train_sp:.2%}')
        print(f'Test  - Loss: {test_loss:.4f}, '
              f'{metric_name}: {test_metric:.4f}')
        print(f'        SpikeRate: {avg_test_sr:.4f}, '
              f'Sparsity: {avg_test_sp:.2%}, '
              f'Spikes/Img: {avg_spk_img:.0f}')
        print(f'Best {metric_name}: {best_val:.4f}, '
              f'LR: {current_lr:.6f}, Time: {epoch_time:.2f}s')
        print('=' * 80)

    # ======================== 保存最终模型 ========================
    final_ckpt = {
        'epoch': args.epochs - 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'max_test_acc': max_test_acc,
        'min_test_mae': min_test_mae,
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

    if writer:
        writer.close()

    best_val = max_test_acc if is_discrete else min_test_mae
    metric_name = 'Acc' if is_discrete else 'MAE'
    print(f'\n训练完成! [{exp_name}] 最佳 {metric_name}: {best_val:.4f}')


if __name__ == '__main__':
    main()
