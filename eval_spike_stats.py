"""
SNN 脉冲统计分析工具
================================
对训练好的模型进行详细的脉冲行为分析，输出：
  1. 全网平均发放率 (target ≈ 0.1)
  2. 分层发放率 (stem / layer1 / layer2 / layer3)
  3. 全网 Sparsity (target > 90%)
  4. 每张图 spike count (能耗 proxy)
  5. FPS 测量
  6. 可选：导出 CSV 用于论文作图

示例:
    # 单模型分析
    python eval_spike_stats.py -weights checkpoint/CIFAR-10/APLIF_ADD_T8/best_model.ckpt

    # 批量对比 (输出对比表)
    python eval_spike_stats.py --batch_compare \\
        checkpoint/CIFAR-10/APLIF_ADD_T8/best_model.ckpt \\
        checkpoint/CIFAR-10/LIF_ADD_T8/best_model.ckpt \\
        checkpoint/CIFAR-10/APLIF_standard_T8/best_model.ckpt

    # 导出 CSV
    python eval_spike_stats.py -weights model.ckpt --export_csv results/stats.csv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import BaseNode
import torch.utils.data as data
import time
import os
import argparse
import numpy as np
import csv
from tqdm import tqdm
from ADD_ResNet110 import resnet110
from collections import OrderedDict

_seed_ = 42
torch.manual_seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


class DetailedSpikeMonitor:
    """详细的脉冲监控器，收集逐层、逐样本的 spike 统计"""

    def __init__(self):
        self.handles = []
        # 逐层统计: name -> {spikes, elements}
        self.layer_spikes = {}
        self.layer_elements = {}
        # 逐神经元统计
        self.neuron_names = []

    def register(self, net: nn.Module):
        for name, module in net.named_modules():
            if isinstance(module, BaseNode):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
                self.neuron_names.append(name)

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.layer_spikes[name] = output.sum().item()
                self.layer_elements[name] = output.numel()
        return hook

    def reset(self):
        self.layer_spikes.clear()
        self.layer_elements.clear()

    def get_per_layer_stats(self):
        """返回逐神经元层的统计"""
        stats = OrderedDict()
        for name in self.neuron_names:
            if name in self.layer_spikes:
                s = self.layer_spikes[name]
                e = self.layer_elements[name]
                stats[name] = {
                    'spike_rate': s / e if e > 0 else 0.0,
                    'sparsity': 1.0 - (s / e) if e > 0 else 1.0,
                    'spikes': s,
                    'elements': e,
                }
        return stats

    def get_group_stats(self):
        """返回分组统计 (stem, layer1, layer2, layer3)"""
        groups = OrderedDict()
        for name in self.neuron_names:
            if name not in self.layer_spikes:
                continue
            parts = name.split('.')
            key = parts[0] if parts[0].startswith('layer') else 'stem'
            if key not in groups:
                groups[key] = {'spikes': 0.0, 'elements': 0}
            groups[key]['spikes'] += self.layer_spikes[name]
            groups[key]['elements'] += self.layer_elements[name]

        result = OrderedDict()
        for k, v in groups.items():
            e = v['elements']
            s = v['spikes']
            result[k] = {
                'spike_rate': s / e if e > 0 else 0.0,
                'sparsity': 1.0 - (s / e) if e > 0 else 1.0,
                'spikes': s,
                'elements': e,
            }
        return result

    def get_total_spike_count(self):
        return sum(self.layer_spikes.values())

    def get_total_elements(self):
        return sum(self.layer_elements.values())

    def get_avg_spike_rate(self):
        total_s = sum(self.layer_spikes.values())
        total_e = sum(self.layer_elements.values())
        return total_s / total_e if total_e > 0 else 0.0

    def get_sparsity(self):
        return 1.0 - self.get_avg_spike_rate()

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def evaluate_model(weights_path, device='cuda:0', batch_size=64,
                   data_dir='./data/CIFAR-10', neuron_type=None,
                   residual_mode=None, T=None):
    """
    评估单个模型，返回完整统计 dict。
    """
    # 加载 checkpoint
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)

    # 读取 config
    if isinstance(ckpt, dict) and 'config' in ckpt:
        cfg = ckpt['config']
        _nt = cfg.get('neuron_type', 'APLIF')
        _rm = cfg.get('residual_mode', 'ADD')
        _T = cfg.get('T', 8)
        _nc = cfg.get('num_classes', 10)
    else:
        _nt = neuron_type or 'APLIF'
        _rm = residual_mode or 'ADD'
        _T = T or 8
        _nc = 10

    # 命令行覆盖
    if neuron_type: _nt = neuron_type
    if residual_mode: _rm = residual_mode
    if T: _T = T

    exp_name = f"{_nt}_{_rm}_T{_T}"

    # 构建网络
    net = resnet110(num_classes=_nc, T=_T, neuron_type=_nt,
                    residual_mode=_rm)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        net.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
    else:
        net.load_state_dict(ckpt)
        epoch = '?'
    net.to(device)
    net.eval()

    # Monitor
    monitor = DetailedSpikeMonitor()
    monitor.register(net)

    # 数据集
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
        ]),
        download=True
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False)

    # 推理
    correct = 0
    total = 0
    total_loss = 0.0
    total_spikes = 0
    all_group_stats = {}
    per_image_spikes = []

    t_start = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Eval [{exp_name}]'):
            monitor.reset()
            images = images.float().to(device)
            labels = labels.to(device)

            out_fr = net(images)
            label_onehot = F.one_hot(labels, _nc).float()
            loss = F.mse_loss(out_fr, label_onehot)

            total_loss += loss.item() * labels.numel()
            _, preds = torch.max(out_fr, 1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            batch_spikes = monitor.get_total_spike_count()
            total_spikes += batch_spikes
            per_image_spikes.append(batch_spikes / labels.numel())

            # 累计分组统计
            for gname, gstat in monitor.get_group_stats().items():
                if gname not in all_group_stats:
                    all_group_stats[gname] = {'spikes': 0.0, 'elements': 0}
                all_group_stats[gname]['spikes'] += gstat['spikes']
                all_group_stats[gname]['elements'] += gstat['elements']

            functional.reset_net(net)

    t_end = time.time()

    accuracy = correct / total
    avg_loss = total_loss / total
    fps = total / (t_end - t_start)
    avg_spikes_per_img = total_spikes / total
    overall_rate = sum(v['spikes'] for v in all_group_stats.values()) / \
                   max(sum(v['elements'] for v in all_group_stats.values()), 1)

    # 分组发放率
    group_rates = OrderedDict()
    for gname in sorted(all_group_stats.keys()):
        v = all_group_stats[gname]
        group_rates[gname] = v['spikes'] / v['elements'] \
            if v['elements'] > 0 else 0.0

    monitor.remove_hooks()
    del net
    torch.cuda.empty_cache()

    return {
        'exp_name': exp_name,
        'weights': weights_path,
        'epoch': epoch,
        'neuron_type': _nt,
        'residual_mode': _rm,
        'T': _T,
        'accuracy': accuracy,
        'loss': avg_loss,
        'avg_spike_rate': overall_rate,
        'sparsity': 1.0 - overall_rate,
        'spikes_per_image': avg_spikes_per_img,
        'fps': fps,
        'group_rates': group_rates,
        'spike_std': np.std(per_image_spikes),
    }


def print_report(result):
    """打印单模型报告"""
    print('\n' + '=' * 70)
    print(f'  脉冲统计报告  [{result["exp_name"]}]')
    print('=' * 70)
    print(f'  权重文件:         {result["weights"]}')
    print(f'  训练 Epoch:       {result["epoch"]}')
    print(f'  配置:             neuron={result["neuron_type"]}, '
          f'residual={result["residual_mode"]}, T={result["T"]}')
    print('-' * 70)
    print(f'  准确率:           {result["accuracy"]:.4f} '
          f'({result["accuracy"]*100:.2f}%)')
    print(f'  测试损失:         {result["loss"]:.4f}')
    print(f'  全网平均发放率:   {result["avg_spike_rate"]:.4f}')
    print(f'  全网 Sparsity:    {result["sparsity"]:.2%}')
    print(f'  每张图 Spikes:    {result["spikes_per_image"]:.0f} '
          f'(± {result["spike_std"]:.0f})')
    print(f'  FPS:              {result["fps"]:.1f} img/s')
    print()
    print('  分层发放率:')
    for gname, grate in result['group_rates'].items():
        bar = '█' * int(grate * 100) + '░' * (20 - int(grate * 100))
        print(f'    {gname:10s}: {grate:.4f}  {bar}')
    print('=' * 70)


def print_comparison_table(results):
    """打印多模型对比表"""
    print('\n' + '=' * 100)
    print('  消融实验对比表')
    print('=' * 100)

    header = (f'{"Config":<25s} {"Acc":>8s} {"Loss":>8s} '
              f'{"SpikeRate":>10s} {"Sparsity":>10s} '
              f'{"Spk/Img":>10s} {"FPS":>8s}')
    print(header)
    print('-' * 100)

    for r in results:
        line = (f'{r["exp_name"]:<25s} '
                f'{r["accuracy"]:>8.4f} {r["loss"]:>8.4f} '
                f'{r["avg_spike_rate"]:>10.4f} {r["sparsity"]:>9.2%} '
                f'{r["spikes_per_image"]:>10.0f} {r["fps"]:>8.1f}')
        print(line)

    print('=' * 100)

    # 分层对比
    print('\n' + '=' * 100)
    print('  分层发放率对比')
    print('=' * 100)
    all_groups = set()
    for r in results:
        all_groups.update(r['group_rates'].keys())
    all_groups = sorted(all_groups)

    header2 = f'{"Config":<25s}'
    for g in all_groups:
        header2 += f' {g:>10s}'
    print(header2)
    print('-' * 100)

    for r in results:
        line2 = f'{r["exp_name"]:<25s}'
        for g in all_groups:
            rate = r['group_rates'].get(g, 0.0)
            line2 += f' {rate:>10.4f}'
        print(line2)
    print('=' * 100)


def main():
    parser = argparse.ArgumentParser(
        description='SNN 脉冲统计分析工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-weights', type=str, default=None,
                        help='单模型 checkpoint 路径')
    parser.add_argument('--batch_compare', nargs='+', default=None,
                        help='多模型对比：提供多个 checkpoint 路径')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-data_dir', type=str, default='./data/CIFAR-10',
                        help='CIFAR-10 数据目录')
    parser.add_argument('--export_csv', type=str, default=None,
                        help='导出 CSV 路径')
    # 手动覆盖 (当 checkpoint 无 config 时)
    parser.add_argument('--neuron_type', default=None, type=str)
    parser.add_argument('--residual_mode', default=None, type=str)
    parser.add_argument('-T', default=None, type=int)

    args = parser.parse_args()

    results = []

    if args.batch_compare:
        # 批量对比模式
        for wp in args.batch_compare:
            if not os.path.exists(wp):
                print(f"[!] 跳过不存在的文件: {wp}")
                continue
            r = evaluate_model(
                wp, device=args.device, batch_size=args.b,
                data_dir=args.data_dir,
                neuron_type=args.neuron_type,
                residual_mode=args.residual_mode,
                T=args.T
            )
            results.append(r)
            print_report(r)

        if len(results) > 1:
            print_comparison_table(results)

    elif args.weights:
        # 单模型模式
        r = evaluate_model(
            args.weights, device=args.device, batch_size=args.b,
            data_dir=args.data_dir,
            neuron_type=args.neuron_type,
            residual_mode=args.residual_mode,
            T=args.T
        )
        results.append(r)
        print_report(r)
    else:
        print("请指定 -weights 或 --batch_compare 参数")
        return

    # 导出 CSV
    if args.export_csv and results:
        os.makedirs(os.path.dirname(args.export_csv) or '.', exist_ok=True)
        fieldnames = ['exp_name', 'neuron_type', 'residual_mode', 'T',
                      'accuracy', 'loss', 'avg_spike_rate', 'sparsity',
                      'spikes_per_image', 'fps', 'epoch', 'weights']
        # 添加分层字段
        all_groups = set()
        for r in results:
            all_groups.update(r['group_rates'].keys())
        for g in sorted(all_groups):
            fieldnames.append(f'rate_{g}')

        with open(args.export_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {k: r[k] for k in fieldnames
                       if k in r and not k.startswith('rate_')}
                for g in sorted(all_groups):
                    row[f'rate_{g}'] = r['group_rates'].get(g, 0.0)
                writer.writerow(row)
        print(f'\n[√] CSV 已导出: {args.export_csv}')


if __name__ == '__main__':
    main()
