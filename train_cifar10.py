"""
CIFAR-10 研究级训练基线
================================
支持一键切换 T / neuron_type / residual_mode 进行消融实验。

示例:
    # 默认 (APLIF + ADD + T=8)
    python train_cifar10.py -epochs 151 -enable_tensorboard

    # 消融：神经元类型
    python train_cifar10.py --neuron_type LIF -epochs 151
    python train_cifar10.py --neuron_type PLIF -epochs 151

    # 消融：残差模式
    python train_cifar10.py --residual_mode standard -epochs 151

    # 消融：时间步
    python train_cifar10.py -T 4 -epochs 151
    python train_cifar10.py -T 16 -epochs 151
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

_seed_ = 42
torch.manual_seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


# ============================================================================
# Spike 统计 Hook (可反传版本)
# ============================================================================
class SpikeMonitor:
    """
    收集所有 BaseNode 子类的脉冲输出，支持：
    - 可反传的 spike_rate 正则 (用 tensor 而非 .item())
    - 分层发放率统计
    - sparsity 统计
    """
    def __init__(self):
        self.spike_tensors = {}   # name -> spike tensor (保留计算图)
        self.spike_counts = {}    # name -> (total_spikes, total_elements)
        self.handles = []

    def register(self, net: nn.Module):
        """对网络中所有 BaseNode 子类注册 hook"""
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
                # 用 detach 版本做统计
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
        """返回 tensor 形式的平均发放率 (保留梯度，可反传)"""
        if not self.spike_tensors:
            return None
        rates = [t.mean() for t in self.spike_tensors.values()]
        return torch.stack(rates).mean()

    def get_avg_spike_rate(self):
        """返回 float 形式的平均发放率 (仅统计)"""
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
            # 提取 layer 前缀 (e.g., 'layer1.0.sn1' -> 'layer1')
            parts = name.split('.')
            layer_key = parts[0] if parts[0].startswith('layer') else 'stem'
            if layer_key not in layer_stats:
                layer_stats[layer_key] = [0.0, 0]
            layer_stats[layer_key][0] += spikes
            layer_stats[layer_key][1] += elements

        return {k: v[0] / v[1] if v[1] > 0 else 0.0
                for k, v in layer_stats.items()}

    def get_total_spike_count(self):
        """返回总 spike 数量 (能耗 proxy)"""
        return sum(v[0] for v in self.spike_counts.values())

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def main():
    parser = argparse.ArgumentParser(
        description='CIFAR-10 SNN 研究级训练基线',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # =============== 框架核心参数 ===============
    parser.add_argument('-T', default=8, type=int,
                        help='仿真时间步 (simulating time-steps)')
    parser.add_argument('--neuron_type', default='APLIF', type=str,
                        choices=['LIF', 'PLIF', 'ALIF', 'APLIF'],
                        help='神经元类型')
    parser.add_argument('--residual_mode', default='ADD', type=str,
                        choices=['standard', 'ADD'],
                        help='残差连接模式')

    # =============== 训练参数 ===============
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=8, type=int, help='batch size')
    parser.add_argument('-epochs', default=151, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-channels', default=128, type=int,
                        help='channels of Conv2d in SNN')
    parser.add_argument('-data_dir', type=str, default='./data/CIFAR-10',
                        help='root dir of cifar10')
    parser.add_argument('-out_dir', type=str, default='./checkpoint/CIFAR-10',
                        help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str,
                        help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true',
                        help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, default='Adam',
                        help='use which optimizer. SGD or Adam')
    parser.add_argument('-lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float,
                        help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='StepLR', type=str,
                        help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=40, type=int,
                        help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float,
                        help='gamma for StepLR')
    parser.add_argument('-T_max', default=32, type=int,
                        help='T_max for CosineAnnealingLR')
    parser.add_argument('-spike_rate_reg', default=0.001, type=float,
                        help='spike rate regularization coefficient')
    parser.add_argument('-target_rate', default=0.1, type=float,
                        help='target average spike rate')
    parser.add_argument('-enable_tensorboard', action='store_true',
                        help='enable tensorboard logging')
    parser.add_argument('-log_interval', default=50, type=int,
                        help='logging interval in batches')

    args = parser.parse_args()

    # =============== 生成实验标识 ===============
    exp_name = f"{args.neuron_type}_{args.residual_mode}_T{args.T}"
    exp_out_dir = os.path.join(args.out_dir, exp_name)
    os.makedirs(exp_out_dir, exist_ok=True)

    # =============== 打印配置 ===============
    print('=' * 70)
    print(f'  CIFAR-10 SNN 研究级基线  [{exp_name}]')
    print('=' * 70)
    print(f'  神经元: {args.neuron_type} | 残差: {args.residual_mode} | T: {args.T}')
    print(f'  LR: {args.lr} | Batch: {args.b} | Epochs: {args.epochs}')
    print(f'  输出目录: {exp_out_dir}')
    print('=' * 70)

    # =============== CUDA 初始化 ===============
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.cuda.synchronize()
            print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory
            print(f"初始显存: {torch.cuda.memory_allocated(0)/1024**2:.0f}MB / "
                  f"{total_mem/1024**2:.0f}MB")
        except RuntimeError as e:
            print(f"CUDA初始化错误: {e}")
            exit(1)

    # =============== TensorBoard ===============
    writer = None
    if args.enable_tensorboard:
        writer = SummaryWriter(os.path.join(exp_out_dir, 'runs'))

    # =============== 构建网络 ===============
    net = resnet110(
        num_classes=10,
        T=args.T,
        neuron_type=args.neuron_type,
        residual_mode=args.residual_mode,
    )
    print(net)
    net.to(args.device)

    # =============== Spike Monitor ===============
    spike_monitor = SpikeMonitor()
    spike_monitor.register(net)

    # =============== 优化器 & 调度器 ===============
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
            optimizer, step_size=args.step_size, gamma=args.gamma)

    # =============== 恢复训练 ===============
    start_epoch = 1
    max_test_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> 从checkpoint恢复: {args.resume}")
            ckpt = torch.load(args.resume, map_location='cpu',
                              weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                net.load_state_dict(ckpt['model_state_dict'])
                start_epoch = ckpt.get('epoch', 0) + 1
                max_test_acc = ckpt.get('max_test_acc', 0.0)
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'lr_scheduler_state_dict' in ckpt:
                    lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
                print(f"=> 从epoch {ckpt['epoch']}恢复, 最佳: {max_test_acc:.4f}")
            else:
                net.load_state_dict(ckpt)
                print("=> 加载仅权重文件")
        else:
            print(f"=> 警告: {args.resume} 不存在，从头训练")

    # =============== 数据集 ===============
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
        ]),
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
        ]),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset, batch_size=args.b,
        shuffle=True, drop_last=True, num_workers=args.j)
    test_data_loader = data.DataLoader(
        dataset=test_dataset, batch_size=args.b,
        shuffle=False, drop_last=False, num_workers=args.j)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    # 类别加权损失
    class_weights = torch.FloatTensor([
        1.0, 1.0, 1.2, 2.0, 1.1, 1.5, 1.0, 1.0, 1.0, 1.0
    ]).to(args.device)

    # =============== 训练循环 ===============
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        start_time = time.time()
        net.train()

        train_loss = 0
        train_cls_loss = 0
        train_reg_loss = 0
        train_acc = 0
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
            label_onehot = F.one_hot(label, 10).float()

            if args.amp:
                with amp.autocast():
                    out_fr = net(frame)
                    mse_loss = F.mse_loss(out_fr, label_onehot,
                                          reduction='none')
                    cls_loss = (mse_loss * class_weights.unsqueeze(0)).mean()

                    # ✅ 可反传的 spike rate 正则
                    sr_tensor = spike_monitor.get_avg_spike_rate_tensor()
                    if sr_tensor is not None:
                        reg_loss = args.spike_rate_reg * (
                            sr_tensor - args.target_rate).square()
                    else:
                        reg_loss = torch.tensor(0.0, device=args.device)
                    loss = cls_loss + reg_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame)
                mse_loss = F.mse_loss(out_fr, label_onehot, reduction='none')
                cls_loss = (mse_loss * class_weights.unsqueeze(0)).mean()

                # ✅ 可反传的 spike rate 正则
                sr_tensor = spike_monitor.get_avg_spike_rate_tensor()
                if sr_tensor is not None:
                    reg_loss = args.spike_rate_reg * (
                        sr_tensor - args.target_rate).square()
                else:
                    reg_loss = torch.tensor(0.0, device=args.device)
                loss = cls_loss + reg_loss

                loss.backward()
                optimizer.step()

            # 统计
            avg_sr = spike_monitor.get_avg_spike_rate()
            sparsity = spike_monitor.get_sparsity()
            epoch_spike_rates.append(avg_sr)
            epoch_sparsities.append(sparsity)

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_cls_loss += cls_loss.item() * label.numel()
            reg_val = reg_loss.item() if torch.is_tensor(reg_loss) else reg_loss
            train_reg_loss += reg_val * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_acc/train_samples:.4f}',
                'sr': f'{avg_sr:.4f}',
                'sp': f'{sparsity:.2%}'
            })

            # TensorBoard batch 日志
            if writer and batch_idx % args.log_interval == 0:
                gs = epoch * len(train_data_loader) + batch_idx
                writer.add_scalar('Train/BatchLoss', loss.item(), gs)
                writer.add_scalar('Train/BatchAcc',
                    (out_fr.argmax(1) == label).float().mean().item(), gs)
                writer.add_scalar('Train/BatchSpikeRate', avg_sr, gs)
                writer.add_scalar('Train/BatchSparsity', sparsity, gs)

            functional.reset_net(net)

        train_loss /= train_samples
        train_cls_loss /= train_samples
        train_reg_loss /= train_samples
        train_acc /= train_samples
        avg_train_sr = np.mean(epoch_spike_rates) if epoch_spike_rates else 0.0
        avg_train_sp = np.mean(epoch_sparsities) if epoch_sparsities else 0.0

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # =============== 测试 ===============
        net.eval()
        test_loss = 0
        test_acc = 0
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
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(frame)
                loss = F.mse_loss(out_fr, label_onehot)

                sr = spike_monitor.get_avg_spike_rate()
                sp = spike_monitor.get_sparsity()
                test_spike_rates.append(sr)
                test_sparsities.append(sp)
                test_total_spikes += spike_monitor.get_total_spike_count()

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()

                test_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{test_acc/test_samples:.4f}',
                    'sr': f'{sr:.4f}'
                })

                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples
        avg_test_sr = np.mean(test_spike_rates) if test_spike_rates else 0.0
        avg_test_sp = np.mean(test_sparsities) if test_sparsities else 0.0
        avg_spikes_per_img = test_total_spikes / test_samples

        epoch_time = time.time() - start_time

        # =============== TensorBoard Epoch 日志 ===============
        if writer:
            writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            writer.add_scalar('Epoch/TrainClsLoss', train_cls_loss, epoch)
            writer.add_scalar('Epoch/TrainRegLoss', train_reg_loss, epoch)
            writer.add_scalar('Epoch/TrainAcc', train_acc, epoch)
            writer.add_scalar('Epoch/TestLoss', test_loss, epoch)
            writer.add_scalar('Epoch/TestAcc', test_acc, epoch)
            writer.add_scalar('Epoch/TrainSpikeRate', avg_train_sr, epoch)
            writer.add_scalar('Epoch/TestSpikeRate', avg_test_sr, epoch)
            writer.add_scalar('Epoch/TrainSparsity', avg_train_sp, epoch)
            writer.add_scalar('Epoch/TestSparsity', avg_test_sp, epoch)
            writer.add_scalar('Epoch/SpikesPerImage', avg_spikes_per_img, epoch)
            writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
            writer.add_scalar('Epoch/Time', epoch_time, epoch)

            # 分层发放率
            spike_monitor.reset()
            # 再跑一个 batch 收集分层信息
            with torch.no_grad():
                sample_frame, _ = next(iter(test_data_loader))
                sample_frame = sample_frame.float().to(args.device)
                net(sample_frame)
                for lname, lrate in spike_monitor.get_layer_rates().items():
                    writer.add_scalar(f'LayerRate/{lname}', lrate, epoch)
                functional.reset_net(net)

        # =============== 保存模型 ===============
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        config = {
            'neuron_type': args.neuron_type,
            'residual_mode': args.residual_mode,
            'T': args.T,
            'num_classes': 10,
            'exp_name': exp_name,
        }

        if save_max:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'max_test_acc': max_test_acc,
                'config': config,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(exp_out_dir, 'best_model.ckpt'))
            torch.save(net.state_dict(),
                       os.path.join(exp_out_dir, 'best_weights.pth'))
            print(f'>>> Saved BEST [{exp_name}] epoch={epoch} '
                  f'acc={test_acc:.4f}')

        if epoch % 10 == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'max_test_acc': max_test_acc,
                'config': config,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(
                exp_out_dir, f'checkpoint_epoch_{epoch}.ckpt'))

        # =============== 打印 ===============
        print('=' * 80)
        print(f'Epoch {epoch}/{args.epochs-1}  [{exp_name}]')
        print(f'Train - Loss: {train_loss:.4f} '
              f'(Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.6f}), '
              f'Acc: {train_acc:.4f}')
        print(f'        SpikeRate: {avg_train_sr:.4f}, '
              f'Sparsity: {avg_train_sp:.2%}')
        print(f'Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}')
        print(f'        SpikeRate: {avg_test_sr:.4f}, '
              f'Sparsity: {avg_test_sp:.2%}, '
              f'Spikes/Img: {avg_spikes_per_img:.0f}')
        print(f'Max Acc: {max_test_acc:.4f}, LR: {current_lr:.6f}, '
              f'Time: {epoch_time:.2f}s')
        print('=' * 80)

    # =============== 保存最终模型 ===============
    final_ckpt = {
        'epoch': args.epochs - 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'max_test_acc': max_test_acc,
        'config': config,
        'args': vars(args),
    }
    torch.save(final_ckpt, os.path.join(exp_out_dir, 'final_model.ckpt'))
    print(f'>>> Saved final model [{exp_name}]')

    if writer:
        writer.close()

    print(f'\n训练完成! [{exp_name}] 最佳测试准确率: {max_test_acc:.4f}')


if __name__ == '__main__':
    main()
