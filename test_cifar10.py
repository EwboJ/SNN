"""
CIFAR-10 SNN 论文消融评测脚本
================================
自动从 checkpoint 读取 config (neuron_type, residual_mode, T)​，构建匹配网络。
输出：准确率 + 混淆矩阵 + spike stats（发放率/稀疏性/分层统计）。
支持加载 full checkpoint（.ckpt）或 weights-only（.pth）。

消融示例:
    # T=4/8/16 时间步消融
    python test_cifar10.py -weights checkpoint/CIFAR-10/APLIF_ADD_T4/best_model.ckpt
    python test_cifar10.py -weights checkpoint/CIFAR-10/APLIF_ADD_T8/best_model.ckpt
    python test_cifar10.py -weights checkpoint/CIFAR-10/APLIF_ADD_T16/best_model.ckpt

    # 神经元消融: LIF vs APLIF
    python test_cifar10.py -weights checkpoint/CIFAR-10/LIF_ADD_T8/best_model.ckpt
    python test_cifar10.py -weights checkpoint/CIFAR-10/APLIF_ADD_T8/best_model.ckpt

    # 残差消融: standard vs ADD
    python test_cifar10.py -weights checkpoint/CIFAR-10/APLIF_standard_T8/best_model.ckpt
    python test_cifar10.py -weights checkpoint/CIFAR-10/APLIF_ADD_T8/best_model.ckpt

    # 加载 weights-only (手动指定配置)
    python test_cifar10.py -weights model.pth --neuron_type APLIF --residual_mode ADD -T 8

    # 批量对比 (用 eval_spike_stats.py)
    python eval_spike_stats.py --batch_compare \\
        checkpoint/CIFAR-10/APLIF_ADD_T8/best_model.ckpt \\
        checkpoint/CIFAR-10/LIF_ADD_T8/best_model.ckpt \\
        --export_csv results/ablation.csv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import BaseNode
import torch.utils.data as data
import time
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，支持保存图片
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import argparse
import numpy as np
from tqdm import tqdm
from ADD_ResNet110 import resnet110
from eval_spike_stats import evaluate_model as eval_spike_detailed

plt.rc('font', family='Times New Roman')

# seed 将在 main() 中根据命令行参数设置


class SpikeMonitor:
    """收集所有 BaseNode 子类的脉冲输出 (评测用，不需要反传)"""
    def __init__(self):
        self.spike_counts = {}
        self.handles = []

    def register(self, net: nn.Module):
        for name, module in net.named_modules():
            if isinstance(module, BaseNode):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.spike_counts[name] = (
                    output.sum().item(), output.numel()
                )
        return hook

    def reset(self):
        self.spike_counts.clear()

    def get_avg_spike_rate(self):
        if not self.spike_counts:
            return 0.0
        total_s = sum(v[0] for v in self.spike_counts.values())
        total_e = sum(v[1] for v in self.spike_counts.values())
        return total_s / total_e if total_e > 0 else 0.0

    def get_sparsity(self):
        return 1.0 - self.get_avg_spike_rate()

    def get_layer_rates(self):
        layer_stats = {}
        for name, (spikes, elements) in self.spike_counts.items():
            parts = name.split('.')
            layer_key = parts[0] if parts[0].startswith('layer') else 'stem'
            if layer_key not in layer_stats:
                layer_stats[layer_key] = [0.0, 0]
            layer_stats[layer_key][0] += spikes
            layer_stats[layer_key][1] += elements
        return {k: v[0]/v[1] if v[1] > 0 else 0.0
                for k, v in layer_stats.items()}

    def get_total_spike_count(self):
        return sum(v[0] for v in self.spike_counts.values())

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def main():
    parser = argparse.ArgumentParser(
        description='CIFAR-10 SNN 论文消融评测',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 框架参数 (可被 checkpoint config 覆盖)
    parser.add_argument('-T', default=8, type=int, help='时间步')
    parser.add_argument('--neuron_type', default='APLIF', type=str,
                        choices=['LIF', 'PLIF', 'ALIF', 'APLIF'],
                        help='神经元类型')
    parser.add_argument('--residual_mode', default='ADD', type=str,
                        choices=['standard', 'ADD'],
                        help='残差模式')
    parser.add_argument('--seed', default=42, type=int,
                        help='随机种子')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-data_dir', type=str, default='./data/CIFAR-10',
                        help='CIFAR-10 数据目录')
    parser.add_argument('-out_dir', type=str, default='./checkpoint/CIFAR-10',
                        help='输出目录')
    parser.add_argument('-weights', type=str, default=None,
                        help='checkpoint 路径 (.ckpt 或 .pth)')
    parser.add_argument('--save_cm', action='store_true',
                        help='保存混淆矩阵图而非显示')
    parser.add_argument('--no_cm', action='store_true',
                        help='不生成混淆矩阵')
    parser.add_argument('--detailed_spike_stats', action='store_true',
                        help='调用 eval_spike_stats 输出详细脉冲统计')

    args = parser.parse_args()

    # =============== 设置随机种子 ===============
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # =============== 加载权重并自动读取 config ===============
    if args.weights is None:
        # 尝试默认路径
        default_path = os.path.join(args.out_dir,
            f'{args.neuron_type}_{args.residual_mode}_T{args.T}',
            'best_model.ckpt')
        if os.path.exists(default_path):
            args.weights = default_path
        else:
            old_path = os.path.join(args.out_dir, 'OA110_confusion.ckpt')
            if os.path.exists(old_path):
                args.weights = old_path
            else:
                print(f"[×] 未找到默认权重文件")
                return

    if not os.path.exists(args.weights):
        print(f"[×] 权重文件不存在: {args.weights}")
        return

    print(f"[√] 加载权重: {args.weights}")
    ckpt_data = torch.load(args.weights, map_location='cpu', weights_only=False)

    # 自动从 checkpoint 读取 config（支持 full checkpoint 和 weights-only）
    is_full_ckpt = isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data
    if isinstance(ckpt_data, dict) and 'config' in ckpt_data:
        cfg = ckpt_data['config']
        neuron_type = cfg.get('neuron_type', args.neuron_type)
        residual_mode = cfg.get('residual_mode', args.residual_mode)
        T = cfg.get('T', args.T)
        num_classes = cfg.get('num_classes', 10)
        print(f"[√] 自动从 checkpoint 读取 config: "
              f"neuron={neuron_type}, residual={residual_mode}, T={T}")
    else:
        neuron_type = args.neuron_type
        residual_mode = args.residual_mode
        T = args.T
        num_classes = 10
        print(f"[!] Checkpoint 无 config 字段，使用命令行参数: "
              f"neuron={neuron_type}, residual={residual_mode}, T={T}")

    exp_name = f"{neuron_type}_{residual_mode}_T{T}"

    # =============== 构建网络 ===============
    net = resnet110(
        num_classes=num_classes,
        T=T,
        neuron_type=neuron_type,
        residual_mode=residual_mode,
    )

    if is_full_ckpt:
        net.load_state_dict(ckpt_data['model_state_dict'])
        epoch_info = ckpt_data.get('epoch', '?')
        best_acc = ckpt_data.get('best_acc',
                                 ckpt_data.get('max_test_acc', '?'))
        print(f"Loaded full checkpoint: epoch={epoch_info}, best_acc={best_acc}")
    else:
        # weights-only (.pth)
        state_dict = ckpt_data if not isinstance(ckpt_data, dict) else ckpt_data
        net.load_state_dict(state_dict)
        print("Loaded weights-only file")

    net.to(args.device)
    net.eval()

    # =============== Spike Monitor ===============
    spike_monitor = SpikeMonitor()
    spike_monitor.register(net)

    # =============== 数据集 ===============
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
        ]),
        download=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset, batch_size=args.b,
        shuffle=False, drop_last=False)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # =============== 推理 ===============
    test_loss = 0
    test_acc = 0
    test_samples = 0
    all_preds = []
    all_labels = []
    all_spike_rates = []
    all_sparsities = []
    total_spikes = 0
    all_layer_rates = {}

    start_time = time.time()
    with torch.no_grad():
        for frame, label in tqdm(test_data_loader, desc=f'Test [{exp_name}]'):
            spike_monitor.reset()
            frame = frame.float().to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()
            out_fr = net(frame)
            loss = F.mse_loss(out_fr, label_onehot)

            sr = spike_monitor.get_avg_spike_rate()
            sp = spike_monitor.get_sparsity()
            all_spike_rates.append(sr)
            all_sparsities.append(sp)
            total_spikes += spike_monitor.get_total_spike_count()

            # 累计分层发放率
            for lname, lrate in spike_monitor.get_layer_rates().items():
                if lname not in all_layer_rates:
                    all_layer_rates[lname] = []
                all_layer_rates[lname].append(lrate)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            _, preds = torch.max(out_fr, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)

    total_time = time.time() - start_time
    test_loss /= test_samples
    test_acc /= test_samples
    fps = test_samples / total_time

    # =============== 输出报告 ===============
    print('\n' + '=' * 70)
    print(f'  评测报告  [{exp_name}]')
    print('=' * 70)
    print(f'  准确率:           {test_acc:.4f} ({test_acc*100:.2f}%)')
    print(f'  测试损失:         {test_loss:.4f}')
    print(f'  平均发放率:       {np.mean(all_spike_rates):.4f}')
    print(f'  全网 Sparsity:    {np.mean(all_sparsities):.2%}')
    print(f'  每张图 Spikes:    {total_spikes/test_samples:.0f}')
    print(f'  FPS:              {fps:.1f} img/s')
    print(f'  推理总时间:       {total_time:.2f}s')
    print()
    print('  分层发放率:')
    for lname in sorted(all_layer_rates.keys()):
        avg_rate = np.mean(all_layer_rates[lname])
        print(f'    {lname:15s}: {avg_rate:.4f}')
    print('=' * 70)

    # =============== 混淆矩阵 ===============
    if not args.no_cm:
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8), dpi=120)
        sns.set(font_scale=1.5)
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Purples",
                    annot_kws={"size": 18}, xticklabels=class_names,
                    yticklabels=class_names)
        plt.title(f'Test Confusion Matrix [{exp_name}]',
                  fontdict={'family': 'Times New Roman', 'size': 18})
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if args.save_cm:
            cm_path = os.path.join(
                args.out_dir,
                f'{neuron_type}_{residual_mode}_T{T}',
                'confusion_matrix.png'
            )
            os.makedirs(os.path.dirname(cm_path), exist_ok=True)
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            print(f'混淆矩阵已保存: {cm_path}')
        else:
            plt.show()

    # =============== 保存报告为 txt ===============
    report_dir = os.path.join(args.out_dir,
                              f'{neuron_type}_{residual_mode}_T{T}')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'eval_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f'实验: {exp_name}\n')
        f.write(f'准确率: {test_acc:.4f}\n')
        f.write(f'损失: {test_loss:.4f}\n')
        f.write(f'平均发放率: {np.mean(all_spike_rates):.4f}\n')
        f.write(f'Sparsity: {np.mean(all_sparsities):.4f}\n')
        f.write(f'Spikes/Image: {total_spikes/test_samples:.0f}\n')
        f.write(f'FPS: {fps:.1f}\n')
        f.write('\n分层发放率:\n')
        for lname in sorted(all_layer_rates.keys()):
            f.write(f'  {lname}: {np.mean(all_layer_rates[lname]):.4f}\n')
    print(f'报告已保存: {report_path}')

    # =============== 详细脉冲统计 (eval_spike_stats) ===============
    if args.detailed_spike_stats:
        print('\n--- 调用 eval_spike_stats 进行详细分析 ---')
        try:
            detailed = eval_spike_detailed(
                args.weights, device=args.device, batch_size=args.b,
                data_dir=args.data_dir,
                neuron_type=neuron_type,
                residual_mode=residual_mode,
                T=T
            )
            from eval_spike_stats import print_report
            print_report(detailed)
        except Exception as e:
            print(f'[!] eval_spike_stats 调用失败: {e}')
            print('    请确保 eval_spike_stats.py 存在且可用')


if __name__ == '__main__':
    main()