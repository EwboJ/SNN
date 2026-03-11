"""
CIFAR-10/100 训练模型可视化评估脚本
=====================================
从训练好的 SorResNet 模型生成：
  1. 混淆矩阵 (confusion_matrix.png + .csv)
  2. 训练 Loss/Acc 曲线 (从 TensorBoard events 提取, 可选)
  3. Spike Rate 分层分析图
  4. 每类 Precision/Recall/F1 表格
  5. metrics.json 汇总文件

支持 CIFAR-10 (10类) 和 CIFAR-100 (100类)。

用法:
    # CIFAR-10, checkpoint 目录下有 final_model.ckpt
    python scripts/plot_cifar_results.py \
        --ckpt ./checkpoint/CIFAR-10/final_model.ckpt \
        --data_dir ./data/CIFAR-10 \
        --dataset cifar10 \
        --out_dir results/cifar10_APLIF_ADD_T8

    # 同时绘制训练曲线 (需要 TensorBoard 目录)
    python scripts/plot_cifar_results.py \
        --ckpt ./checkpoint/CIFAR-10/final_model.ckpt \
        --data_dir ./data/CIFAR-10 \
        --tb_dir ./checkpoint/CIFAR-10/runs \
        --out_dir results/cifar10_APLIF_ADD_T8
"""

import os
import sys
import json
import csv
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF
import torch.utils.data as data
from tqdm import tqdm

# 路径处理：找到项目根目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT_DIR)

from ADD_ResNet110 import resnet110
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import BaseNode

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------- 可选依赖 ----------
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False

try:
    from sklearn.metrics import precision_recall_fscore_support
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------- 全局样式 ----------
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
for font in ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']:
    try:
        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

# ---------- CIFAR 类别名 ----------
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
# CIFAR-100 fine labels (20 superclass × 5 fine)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
    'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
    'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
    'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
    'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
    'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
    'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
    'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
    'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
    'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


# ============================================================================
# SpikeMonitor
# ============================================================================
class SpikeMonitor:
    def __init__(self):
        self.layer_spikes = {}
        self.layer_elements = {}
        self.handles = []

    def register(self, net):
        for name, m in net.named_modules():
            if isinstance(m, BaseNode):
                self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(mod, inp, out):
            if isinstance(out, torch.Tensor):
                self.layer_spikes[name] = out.sum().item()
                self.layer_elements[name] = out.numel()
        return fn

    def reset(self):
        self.layer_spikes.clear()
        self.layer_elements.clear()

    def avg_rate(self):
        s = sum(self.layer_spikes.values())
        e = sum(self.layer_elements.values())
        return s / e if e > 0 else 0.0

    def total_spikes(self):
        return sum(self.layer_spikes.values())

    def group_rates(self):
        groups = OrderedDict()
        for name in sorted(self.layer_spikes.keys()):
            parts = name.split('.')
            key = parts[0] if parts[0].startswith('layer') else 'stem'
            if key not in groups:
                groups[key] = [0.0, 0]
            groups[key][0] += self.layer_spikes[name]
            groups[key][1] += self.layer_elements[name]
        return {k: v[0] / v[1] if v[1] > 0 else 0.0
                for k, v in groups.items()}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ============================================================================
# 图表生成函数
# ============================================================================
def plot_confusion_matrix(cm, names, out_dir, accuracy, max_show=20):
    """
    绘制混淆矩阵。
    CIFAR-100 有 100 类，标签会拥挤，限制最多显示 max_show 类的详细标签。
    """
    n = len(names)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig_w = max(12, n * 0.35)
    fig_h = max(5, n * 0.3)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w * 2, fig_h))

    show_labels = n <= max_show
    tick_labels = names if show_labels else [str(i) for i in range(n)]
    fontsize_tick = max(6, 11 - n // 10)
    fontsize_val = max(5, 10 - n // 12)

    for ax, data_mat, cmap, title in [
        (axes[0], cm, 'Blues', 'Confusion Matrix (Count)'),
        (axes[1], cm_norm, 'Oranges',
         f'Confusion Matrix (%) — Acc={accuracy:.2%}'),
    ]:
        im = ax.imshow(data_mat, cmap=cmap, aspect='auto',
                       vmin=0, vmax=(None if ax == axes[0] else 1))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, fontsize=fontsize_tick,
                           rotation=45, ha='right')
        ax.set_yticklabels(tick_labels, fontsize=fontsize_tick)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(title, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046)

        # 只在类别较少时写数值
        if n <= max_show:
            threshold = data_mat.max() / 2
            for i in range(n):
                for j in range(n):
                    v = data_mat[i, j]
                    color = 'white' if v > threshold else 'black'
                    label = str(int(v)) if ax == axes[0] else f'{v*100:.0f}%'
                    ax.text(j, i, label, ha='center', va='center',
                            fontsize=fontsize_val, fontweight='bold',
                            color=color)

    plt.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {path}')

    # CSV
    csv_path = os.path.join(out_dir, 'confusion_matrix.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([''] + names)
        for i, name in enumerate(names):
            w.writerow([name] + cm[i].tolist())
    print(f'  [✓] {csv_path}')


def plot_training_curves(tb_dir, out_dir):
    if not HAS_TB:
        print('  [!] tensorboard 未安装，跳过训练曲线')
        return False

    ea = EventAccumulator(tb_dir)
    ea.Reload()
    available = ea.Tags().get('scalars', [])
    if not available:
        print(f'  [!] 未找到 TensorBoard 标量数据: {tb_dir}')
        return False

    print(f'  可用指标: {available}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax = axes[0]
    for tags, color, label in [
        (['Epoch/TrainLoss'], '#2196F3', 'Train Loss'),
        (['Epoch/ValLoss', 'Epoch/TestLoss'], '#FF5722', 'Val Loss'),
    ]:
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                ax.plot([e.step for e in events], [e.value for e in events],
                        color=color, label=label, linewidth=1.5)
                break
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training / Val Loss')
    ax.legend()

    # Accuracy 曲线
    ax2 = axes[1]
    for tags, color, label in [
        (['Epoch/TrainAcc'], '#4CAF50', 'Train Acc'),
        (['Epoch/ValAcc', 'Epoch/TestAcc'], '#9C27B0', 'Val Acc'),
    ]:
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                ax2.plot([e.step for e in events], [e.value for e in events],
                         color=color, label=label, linewidth=1.5)
                break
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training / Val Accuracy')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {path}')

    # Spike Rate 曲线
    fig2, ax3 = plt.subplots(figsize=(8, 4))
    for tags, color, label in [
        (['Epoch/TrainSpikeRate'], '#FF9800', 'Train Spike Rate'),
        (['Epoch/ValSpikeRate', 'Epoch/TestSpikeRate'], '#E91E63', 'Val Spike Rate'),
    ]:
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                ax3.plot([e.step for e in events], [e.value for e in events],
                         color=color, label=label, linewidth=1.5)
                break
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Spike Rate')
    ax3.set_title('Spike Rate over Training')
    ax3.legend()
    plt.tight_layout()
    path2 = os.path.join(out_dir, 'spike_rate_curve.png')
    fig2.savefig(path2, bbox_inches='tight')
    plt.close(fig2)
    print(f'  [✓] {path2}')

    return True


def plot_spike_analysis(group_rates, spikes_per_img, accuracy, out_dir,
                        exp_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    groups = list(group_rates.keys())
    rates = [group_rates[g] for g in groups]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    bars = ax.bar(groups, rates, color=colors[:len(groups)],
                  edgecolor='white', linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002, f'{rate:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Spike Rate')
    ax.set_title('Layer-wise Spike Rate')
    ax.axhline(y=np.mean(rates), color='red', linestyle='--', alpha=0.5,
               label=f'Avg={np.mean(rates):.4f}')
    ax.legend()

    ax2 = axes[1]
    ax2.axis('off')
    overall_rate = np.mean(rates)
    sparsity = 1.0 - overall_rate
    card_text = (
        f"{'═' * 35}\n"
        f"  实验: {exp_name}\n"
        f"{'─' * 35}\n"
        f"  准确率:         {accuracy:.2%}\n"
        f"  平均发放率:     {overall_rate:.4f}\n"
        f"  稀疏度:         {sparsity:.2%}\n"
        f"  每帧 Spikes:    {spikes_per_img:.0f}\n"
        f"{'═' * 35}"
    )
    ax2.text(0.1, 0.5, card_text, transform=ax2.transAxes,
             fontsize=13, fontfamily='monospace', verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F4F8',
                       edgecolor='#90A4AE', linewidth=1.5))

    plt.tight_layout()
    path = os.path.join(out_dir, 'spike_analysis.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {path}')


def plot_per_class_metrics(per_class, out_dir, max_table_rows=30):
    """
    绘制每类 Precision/Recall/F1 表格。
    类别太多时 (>max_table_rows) 仅保存 CSV，同时绘制 bar 图。
    """
    n = len(per_class)
    names = [p['name'] for p in per_class]
    precisions = [p['precision'] for p in per_class]
    recalls = [p['recall'] for p in per_class]
    f1s = [p['f1'] for p in per_class]

    # 始终保存 CSV
    csv_path = os.path.join(out_dir, 'per_class_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['class', 'precision', 'recall',
                                          'f1', 'support'])
        w.writeheader()
        for p in per_class:
            w.writerow({'class': p['name'], 'precision': f"{p['precision']:.4f}",
                        'recall': f"{p['recall']:.4f}", 'f1': f"{p['f1']:.4f}",
                        'support': p['support']})
    print(f'  [✓] {csv_path}')

    if n <= max_table_rows:
        # 表格图
        fig, ax = plt.subplots(figsize=(9, max(3, n * 0.35)))
        ax.axis('off')
        headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        rows = [[p['name'], f"{p['precision']:.4f}", f"{p['recall']:.4f}",
                 f"{p['f1']:.4f}", str(p['support'])] for p in per_class]
        table = ax.table(cellText=rows, colLabels=headers, loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.4)
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#37474F')
                cell.set_text_props(color='white', fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#ECEFF1')
            cell.set_edgecolor('#B0BEC5')
        plt.tight_layout()
        path = os.path.join(out_dir, 'per_class_metrics.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'  [✓] {path}')
    else:
        # 类别太多，绘制 F1 柱状图
        fig, ax = plt.subplots(figsize=(max(12, n * 0.18), 5))
        x = np.arange(n)
        w = 0.28
        ax.bar(x - w, precisions, w, label='Precision', color='#2196F3',
               alpha=0.85)
        ax.bar(x,     recalls,    w, label='Recall',    color='#4CAF50',
               alpha=0.85)
        ax.bar(x + w, f1s,        w, label='F1',        color='#FF9800',
               alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_ylabel('Score')
        ax.set_title(f'Per-class Metrics ({n} classes)')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, 'per_class_metrics.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'  [✓] {path}')


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='CIFAR-10/100 SNN 模型评估与可视化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ckpt', type=str, required=True,
                        help='模型 checkpoint 路径 (.ckpt 或 .pth)')
    parser.add_argument('--data_dir', type=str, default='./data/CIFAR-10',
                        help='CIFAR 数据集根目录')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='数据集类型')
    parser.add_argument('--tb_dir', type=str, default=None,
                        help='TensorBoard events 目录 (可选, 用于绘制训练曲线)')
    parser.add_argument('--out_dir', type=str, default='./results/cifar_eval',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='推理设备')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='推理 batch size')
    # 允许手动覆盖从 checkpoint config 读取的参数
    parser.add_argument('--neuron_type', type=str, default=None,
                        choices=['LIF', 'PLIF', 'ALIF', 'APLIF'],
                        help='神经元类型 (不指定则从 checkpoint 读取)')
    parser.add_argument('--residual_mode', type=str, default=None,
                        choices=['standard', 'ADD'],
                        help='残差模式 (不指定则从 checkpoint 读取)')
    parser.add_argument('-T', type=int, default=None,
                        help='时间步 (不指定则从 checkpoint 读取)')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 70)
    print('  CIFAR 模型评估与可视化')
    print('=' * 70)

    # ---- 加载 Checkpoint ----
    print('\n[1/5] 加载 checkpoint...')
    device = args.device if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict):
        cfg = ckpt.get('config', {})
        epoch = ckpt.get('epoch', '?')
        max_acc = ckpt.get('max_val_acc', ckpt.get('max_test_acc', None))
        state_dict = ckpt.get('model_state_dict', ckpt)
    else:
        cfg = {}
        epoch = '?'
        max_acc = None
        state_dict = ckpt

    # 优先使用命令行参数，其次读取 checkpoint config，最后用默认值
    neuron_type = args.neuron_type or cfg.get('neuron_type', 'APLIF')
    residual_mode = args.residual_mode or cfg.get('residual_mode', 'ADD')
    T = args.T or cfg.get('T', 8)
    dataset_name = cfg.get('dataset', args.dataset)
    num_classes = cfg.get('num_classes', 10 if 'cifar10' in dataset_name else 100)
    in_channels = cfg.get('in_channels', 3)
    exp_name = cfg.get('exp_name', f'{neuron_type}_{residual_mode}_T{T}')

    print(f'  神经元类型:   {neuron_type}')
    print(f'  残差模式:     {residual_mode}')
    print(f'  时间步 T:     {T}')
    print(f'  数据集:       {dataset_name} ({num_classes} 类)')
    print(f'  Epoch:        {epoch}')
    if max_acc is not None:
        print(f'  Best Val Acc: {max_acc:.4f}')

    # ---- 构建模型 ----
    print('\n[2/5] 构建并加载模型...')
    net = resnet110(
        num_classes=num_classes,
        T=T,
        neuron_type=neuron_type,
        residual_mode=residual_mode,
        in_channels=in_channels,
    )
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    total_params = sum(p.numel() for p in net.parameters())
    print(f'  [✓] 模型加载成功 | 参数量: {total_params/1e6:.2f}M')

    # ---- 构建测试数据集 ----
    print('\n[3/5] 加载测试数据集...')
    if 'cifar10' in dataset_name and '100' not in dataset_name:
        transform = TF.Compose([
            TF.ToTensor(),
            TF.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
        ])
        test_ds = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False,
            transform=transform, download=True)
        class_names = CIFAR10_CLASSES
    else:
        transform = TF.Compose([
            TF.ToTensor(),
            TF.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
        test_ds = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False,
            transform=transform, download=True)
        class_names = CIFAR100_CLASSES

    loader = data.DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    print(f'  [✓] 测试集: {len(test_ds)} 张图像')

    # ---- 推理 ----
    print('\n[4/5] 推理...')
    monitor = SpikeMonitor()
    monitor.register(net)

    all_preds, all_labels = [], []
    total_spikes, total_imgs = 0, 0
    all_spike_rates = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Inference'):
            monitor.reset()
            images = images.to(device)
            out = net(images)

            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

            total_spikes += monitor.total_spikes()
            total_imgs += images.shape[0]
            all_spike_rates.append(monitor.avg_rate())
            functional.reset_net(net)

    group_rates = monitor.group_rates()
    monitor.remove()

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)
    spikes_per_img = total_spikes / max(total_imgs, 1)
    avg_sr = float(np.mean(all_spike_rates))
    sparsity = 1.0 - avg_sr

    print(f'\n  ✓ 准确率:       {accuracy:.4f}  ({accuracy:.2%})')
    print(f'  ✓ 平均发放率:   {avg_sr:.4f}')
    print(f'  ✓ 稀疏度:       {sparsity:.2%}')
    print(f'  ✓ Spikes/Img:   {spikes_per_img:.0f}')

    # ---- 生成图表 ----
    print('\n[5/5] 生成图表...')

    # 5a. 混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(all_preds, all_labels):
        cm[l][p] += 1
    plot_confusion_matrix(cm, class_names, args.out_dir, accuracy)

    # 5b. Per-class 指标
    if HAS_SKLEARN:
        p_arr, r_arr, f1_arr, sup_arr = precision_recall_fscore_support(
            all_labels, all_preds,
            labels=list(range(num_classes)), zero_division=0)
        per_class = [
            {'name': class_names[i],
             'precision': float(p_arr[i]),
             'recall': float(r_arr[i]),
             'f1': float(f1_arr[i]),
             'support': int(sup_arr[i])}
            for i in range(num_classes)
        ]
        plot_per_class_metrics(per_class, args.out_dir)
        macro_f1 = float(np.mean(f1_arr))
    else:
        print('  [!] sklearn 未安装，跳过 per-class 指标')
        per_class = []
        macro_f1 = None

    # 5c. Spike 分析图
    plot_spike_analysis(group_rates, spikes_per_img, accuracy,
                        args.out_dir, exp_name)

    # 5d. 训练曲线 (TensorBoard)
    tb_dir = args.tb_dir
    if not tb_dir:
        ckpt_dir = os.path.dirname(args.ckpt)
        runs_dir = os.path.join(ckpt_dir, 'runs')
        if os.path.isdir(runs_dir):
            tb_dir = runs_dir
        else:
            events = [f for f in os.listdir(ckpt_dir)
                      if f.startswith('events.out.tfevents')]
            if events:
                tb_dir = ckpt_dir
    if tb_dir:
        plot_training_curves(tb_dir, args.out_dir)
    else:
        print('  [!] 未找到 TensorBoard 目录，跳过训练曲线')

    # ---- 保存 metrics.json ----
    metrics = {
        'exp_name': exp_name,
        'dataset': dataset_name,
        'epoch': epoch,
        'neuron_type': neuron_type,
        'residual_mode': residual_mode,
        'T': T,
        'num_classes': num_classes,
        'total_params': total_params,
        'test_accuracy': round(accuracy, 6),
        'avg_spike_rate': round(avg_sr, 6),
        'sparsity': round(sparsity, 6),
        'spikes_per_image': round(spikes_per_img, 1),
        'total_test_images': total_imgs,
        'group_rates': {k: round(v, 6) for k, v in group_rates.items()},
    }
    if macro_f1 is not None:
        metrics['macro_f1'] = round(macro_f1, 6)
    if per_class:
        metrics['per_class'] = per_class
    metrics['confusion_matrix'] = cm.tolist()
    if max_acc is not None:
        metrics['best_val_acc_in_ckpt'] = round(float(max_acc), 6)

    mpath = os.path.join(args.out_dir, 'metrics.json')
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'  [✓] {mpath}')

    # ---- 汇总 ----
    print('\n' + '=' * 70)
    print(f'  评估完成！结果保存至: {os.path.abspath(args.out_dir)}/')
    print('=' * 70)
    for fn in sorted(os.listdir(args.out_dir)):
        fp = os.path.join(args.out_dir, fn)
        sz = os.path.getsize(fp) / 1024
        print(f'    {fn:35s}  {sz:6.1f} KB')
    print('=' * 70)


if __name__ == '__main__':
    main()
