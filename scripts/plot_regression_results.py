"""
回归任务实验可视化脚本
=====================================
专门服务 corridor regression / straight_keep_reg 任务:
  1. prediction_vs_gt.png   — 预测 vs 真值散点 + 理想线
  2. residual_hist.png      — 残差分布直方图
  3. phase_metrics.png      — Correcting / Settled 分段指标 (若有 phase)
  4. spike_analysis.png     — SNN 发放率分层柱状图 + 指标卡片
  5. training_curves.png    — Loss / MAE 训练曲线 (TensorBoard)
  6. metrics.json           — 完整数值指标
  7. predictions.csv        — 逐样本预测结果

用法:
    python scripts/plot_regression_results.py \\
        --ckpt checkpoint/corridor/straight_keep_reg_APLIF_ADD_T4/best_model.ckpt \\
        --data_root ./data/straight_keep/straight_keep_reg_v1/test \\
        --tb_dir checkpoint/corridor/straight_keep_reg_APLIF_ADD_T4/runs \\
        --out_dir results/straight_keep_reg_APLIF_ADD_T4
"""

import os
import sys
import json
import csv
import math
import argparse
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torchvision
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import BaseNode
import torch.utils.data as data
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- 可选依赖 ----------
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from tensorboard.backend.event_processing.event_accumulator \
        import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False

# ---------- 全局样式 ----------
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 中文字体 fallback
for font in ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']:
    try:
        plt.rcParams['font.sans-serif'] = (
            [font] + plt.rcParams['font.sans-serif'])
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# SpikeMonitor (与 plot_results.py 保持一致)
# ============================================================================
class SpikeMonitor:
    def __init__(self):
        self.layer_spikes = {}
        self.layer_elements = {}
        self.handles = []

    def register(self, net):
        for name, m in net.named_modules():
            if isinstance(m, BaseNode):
                self.handles.append(
                    m.register_forward_hook(self._hook(name)))

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
        return s / e if e > 0 else 0

    def total_spikes(self):
        return sum(self.layer_spikes.values())

    def group_rates(self):
        groups = OrderedDict()
        for name in sorted(self.layer_spikes.keys()):
            parts = name.split('.')
            if 'backbone' in name:
                sub = parts[1] if len(parts) > 1 else parts[0]
                key = sub if sub.startswith('layer') else 'stem'
            elif 'head' in name:
                key = 'head'
            else:
                key = (parts[0] if parts[0].startswith('layer')
                       else 'other')
            if key not in groups:
                groups[key] = [0.0, 0]
            groups[key][0] += self.layer_spikes[name]
            groups[key][1] += self.layer_elements[name]
        return {k: v[0] / v[1] if v[1] > 0 else 0
                for k, v in groups.items()}

    def remove(self):
        for h in self.handles:
            h.remove()


# ============================================================================
# 1. Prediction vs Ground Truth
# ============================================================================
def plot_prediction_vs_gt(preds, labels, out_dir, exp_name, mae, rmse,
                         zero_mae=None, zero_rmse=None):
    """预测 vs 真值散点图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 散点
    ax = axes[0]
    ax.scatter(labels, preds, s=8, alpha=0.4, color='#2196F3',
               edgecolors='none')
    lo = min(min(labels), min(preds))
    hi = max(max(labels), max(preds))
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            'r--', linewidth=1.5, alpha=0.7, label='Ideal')
    ax.set_xlabel('Ground Truth (angular_z)', fontsize=12)
    ax.set_ylabel('Prediction', fontsize=12)
    title = (f'Prediction vs GT — {exp_name}\n'
             f'MAE={mae:.4f}  RMSE={rmse:.4f}')
    if zero_mae is not None and zero_rmse is not None:
        title += f'\nZeroBaseline MAE={zero_mae:.4f}  RMSE={zero_rmse:.4f}'
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_aspect('equal', 'box')

    # 右: 时间序列对比 (前 300 帧)
    ax2 = axes[1]
    n_show = min(300, len(preds))
    x = np.arange(n_show)
    ax2.plot(x, labels[:n_show], color='#4CAF50', linewidth=1.2,
             alpha=0.8, label='Ground Truth')
    ax2.plot(x, preds[:n_show], color='#FF5722', linewidth=1.0,
             alpha=0.7, label='Prediction')
    ax2.fill_between(x,
                     np.array(labels[:n_show]) - np.array(preds[:n_show]),
                     0, alpha=0.15, color='#FF5722')
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('angular_z', fontsize=12)
    ax2.set_title('Time-series Comparison (first 300)', fontsize=13)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, 'prediction_vs_gt.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {path}')


# ============================================================================
# 2. Residual Histogram
# ============================================================================
def plot_residual_hist(residuals, out_dir, exp_name):
    """残差分布直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    arr = np.array(residuals)

    # 左: 直方图
    ax = axes[0]
    bins = min(80, max(20, len(arr) // 20))
    if HAS_SEABORN:
        sns.histplot(arr, bins=bins, kde=True, ax=ax, color='#5C6BC0')
    else:
        ax.hist(arr, bins=bins, color='#5C6BC0', edgecolor='white',
                alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.6)
    ax.axvline(x=np.mean(arr), color='orange', linestyle='-.',
               alpha=0.6, label=f'Mean={np.mean(arr):.4f}')
    ax.set_xlabel('Residual (pred - gt)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Residual Distribution — {exp_name}', fontsize=13)
    ax.legend(fontsize=10)

    # 右: Q-Q plot (简易)
    ax2 = axes[1]
    sorted_res = np.sort(arr)
    n = len(sorted_res)
    q = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    from scipy import stats as sp_stats
    try:
        theoretical = sp_stats.norm.ppf(q)
        ax2.scatter(theoretical, sorted_res, s=4, alpha=0.4,
                    color='#26A69A')
        ax2.plot([theoretical[0], theoretical[-1]],
                 [theoretical[0] * np.std(arr) + np.mean(arr),
                  theoretical[-1] * np.std(arr) + np.mean(arr)],
                 'r--', linewidth=1.2, alpha=0.7)
        ax2.set_xlabel('Theoretical Quantiles', fontsize=12)
        ax2.set_ylabel('Sample Quantiles', fontsize=12)
        ax2.set_title('Q-Q Plot', fontsize=13)
    except ImportError:
        ax2.text(0.5, 0.5, 'scipy 未安装\n跳过 Q-Q plot',
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=14)
        ax2.axis('off')

    plt.tight_layout()
    path = os.path.join(out_dir, 'residual_hist.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {path}')


# ============================================================================
# 3. Phase Metrics (Correcting / Settled)
# ============================================================================
def plot_phase_metrics(phase_stats, out_dir, exp_name):
    """Correcting / Settled 分段指标对比"""
    phases = list(phase_stats.keys())
    if not phases:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAE 对比
    ax = axes[0]
    colors = {'Correcting': '#FF7043', 'Settled': '#66BB6A'}
    default_color = '#78909C'
    mae_vals = [phase_stats[p]['mae'] for p in phases]
    bar_colors = [colors.get(p, default_color) for p in phases]
    bars = ax.bar(phases, mae_vals, color=bar_colors, edgecolor='white',
                  linewidth=0.5)
    for bar, v in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:.4f}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Phase MAE', fontsize=13)

    # RMSE 对比
    ax2 = axes[1]
    rmse_vals = [phase_stats[p]['rmse'] for p in phases]
    bars2 = ax2.bar(phases, rmse_vals, color=bar_colors, edgecolor='white',
                    linewidth=0.5)
    for bar, v in zip(bars2, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{v:.4f}', ha='center', va='bottom', fontsize=11,
                 fontweight='bold')
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('Phase RMSE', fontsize=13)

    # 样本数
    ax3 = axes[2]
    counts = [phase_stats[p]['count'] for p in phases]
    bars3 = ax3.bar(phases, counts, color=bar_colors, edgecolor='white',
                    linewidth=0.5)
    for bar, v in zip(bars3, counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(v), ha='center', va='bottom', fontsize=11,
                 fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Phase Sample Count', fontsize=13)

    fig.suptitle(f'Phase-level Metrics — {exp_name}', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'phase_metrics.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {path}')


# ============================================================================
# 4. Spike Analysis
# ============================================================================
def plot_spike_analysis_reg(grp_rates, spikes_per_img, mae, rmse,
                            out_dir, exp_name):
    """SNN 发放率分层分析 (回归版)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 分层发放率
    ax = axes[0]
    groups = list(grp_rates.keys())
    rates = [grp_rates[g] for g in groups]
    palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
               '#FFEAA7', '#DDA0DD']
    bars = ax.bar(groups, rates, color=palette[:len(groups)],
                  edgecolor='white', linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f'{rate:.4f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_ylabel('Spike Rate', fontsize=12)
    ax.set_title('Layer-wise Spike Rate', fontsize=13)
    ax.axhline(y=np.mean(rates), color='red', linestyle='--', alpha=0.5,
               label=f'Avg={np.mean(rates):.4f}')
    ax.legend(fontsize=10)

    # 右: 指标卡片
    ax2 = axes[1]
    ax2.axis('off')
    overall_rate = np.mean(rates)
    sparsity = 1.0 - overall_rate

    card = (
        f"{'═' * 35}\n"
        f"  实验: {exp_name}\n"
        f"{'─' * 35}\n"
        f"  MAE:            {mae:.4f}\n"
        f"  RMSE:           {rmse:.4f}\n"
        f"  平均发放率:     {overall_rate:.4f}\n"
        f"  稀疏度:         {sparsity:.2%}\n"
        f"  每帧 Spikes:    {spikes_per_img:.0f}\n"
        f"{'═' * 35}"
    )
    ax2.text(0.1, 0.5, card, transform=ax2.transAxes,
             fontsize=13, fontfamily='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.8',
                       facecolor='#F0F4F8',
                       edgecolor='#90A4AE', linewidth=1.5))

    plt.tight_layout()
    path = os.path.join(out_dir, 'spike_analysis.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {path}')


# ============================================================================
# 5. Training Curves (TensorBoard)
# ============================================================================
def plot_training_curves_reg(tb_dir, out_dir):
    """从 TensorBoard 提取训练曲线 (回归版)"""
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

    # Loss
    ax = axes[0]
    for tags, color, label in [
        (['Epoch/TrainLoss'], '#2196F3', 'Train Loss'),
        (['Epoch/ValLoss', 'Epoch/TestLoss'], '#FF5722', 'Val Loss'),
    ]:
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                ax.plot(steps, vals, color=color, label=label,
                        linewidth=1.5)
                break
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training / Val Loss', fontsize=13)
    ax.legend(fontsize=11)

    # MAE
    ax2 = axes[1]
    for tags, color, label in [
        (['Epoch/TrainMAE'], '#4CAF50', 'Train MAE'),
        (['Epoch/ValMAE', 'Epoch/TestMAE'], '#9C27B0', 'Val MAE'),
    ]:
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                ax2.plot(steps, vals, color=color, label=label,
                         linewidth=1.5)
                break
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Training / Val MAE', fontsize=13)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {path}')

    # Spike Rate 曲线
    fig2, ax3 = plt.subplots(figsize=(8, 4))
    for tags, color, label in [
        (['Epoch/TrainSpikeRate'], '#FF9800', 'Train Spike Rate'),
        (['Epoch/ValSpikeRate', 'Epoch/TestSpikeRate'],
         '#E91E63', 'Val Spike Rate'),
    ]:
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                ax3.plot(steps, vals, color=color, label=label,
                         linewidth=1.5)
                break
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Spike Rate', fontsize=12)
    ax3.set_title('Spike Rate over Training', fontsize=13)
    ax3.legend(fontsize=11)
    plt.tight_layout()
    path2 = os.path.join(out_dir, 'spike_rate_curve.png')
    fig2.savefig(path2, bbox_inches='tight')
    plt.close(fig2)
    print(f'  [✓] {path2}')
    return True


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='回归任务实验可视化 (straight_keep_reg)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='测试数据路径 (包含 run 目录)')
    parser.add_argument('--tb_dir', type=str, default=None,
                        help='TensorBoard events 目录 (可选)')
    parser.add_argument('--out_dir', type=str,
                        default='./results/regression_eval',
                        help='输出目录')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 70)
    print('  回归任务实验可视化')
    print('=' * 70)

    # ---- [1/6] 加载模型 ----
    print('\n[1/6] 加载模型...')
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}

    dataset_type = cfg.get('dataset', 'corridor')
    mode = cfg.get('mode', 'regression')
    encoding = cfg.get('encoding', 'rate')
    T = cfg.get('T', 4)
    neuron_type = cfg.get('neuron_type', 'APLIF')
    residual_mode = cfg.get('residual_mode', 'ADD')
    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    control_dim = cfg.get('control_dim', 1)
    num_actions = cfg.get('num_classes', 1)
    img_h = cfg.get('img_h', 32)
    img_w = cfg.get('img_w', 32)
    seq_len = cfg.get('seq_len', None)
    stride = cfg.get('stride', None)
    is_sequence = cfg.get('is_sequence', False)

    # 检查: 仅支持 regression
    is_discrete = (mode == 'discrete')
    if is_discrete:
        print(f'  ✗ 此脚本仅支持 regression 模式')
        print(f'    当前 mode={mode}')
        print(f'    离散分类请使用 plot_results.py')
        return

    exp_name = cfg.get('exp_name',
                       f'{neuron_type}_{residual_mode}_T{T}')

    print(f'  Config: {exp_name}')
    print(f'  dataset:    {dataset_type}')
    print(f'  mode:       {mode}')
    print(f'  encoding:   {encoding}')
    print(f'  T:          {T}')
    print(f'  neuron:     {neuron_type}')
    print(f'  residual:   {residual_mode}')
    print(f'  img:        {img_h}x{img_w}')
    print(f'  epoch:      {epoch}')
    if seq_len:
        print(f'  seq_len:    {seq_len}')
        print(f'  stride:     {stride}')

    # 构建网络
    sys.path.insert(
        0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(
        0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..'))
    from models.snn_corridor import build_corridor_net

    net = build_corridor_net(
        head_type=mode, num_actions=num_actions,
        control_dim=control_dim,
        encoding=encoding, T=T, neuron_type=neuron_type,
        residual_mode=residual_mode)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        net.load_state_dict(ckpt['model_state_dict'])
    else:
        net.load_state_dict(ckpt)
    net.to(args.device)
    net.eval()
    print(f'  [✓] 模型加载成功')

    # ---- [2/6] 加载数据集 ----
    print('\n[2/6] 加载测试数据...')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_h, img_w)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # corridor_task dataset (return_meta=True 读取 phase)
    is_corridor_task = (dataset_type == 'corridor_task')
    has_phase = False

    if is_corridor_task:
        from datasets.corridor_task_dataset import CorridorTaskDataset
        test_ds = CorridorTaskDataset(
            root_dir=args.data_root, transforms=transform,
            return_meta=True, print_stats=True)
        # 检查第一个样本是否有 phase
        if len(test_ds) > 0:
            _, _, sample_meta = test_ds[0]
            has_phase = bool(sample_meta.get('phase', ''))
    else:
        from datasets.corridor_dataset import CorridorDataset
        test_ds = CorridorDataset(
            root_dir=args.data_root, mode=mode,
            control_dim=control_dim,
            valid_only=True, transforms=transform)

    loader = data.DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    print(f'  [✓] 测试集: {len(test_ds)} 帧')
    print(f'  [✓] has_phase: {has_phase}')

    # ---- [3/6] 推理 ----
    print('\n[3/6] 推理...')
    monitor = SpikeMonitor()
    monitor.register(net)

    all_preds = []
    all_labels = []
    all_phases = []
    total_spikes = 0
    total_frames = 0
    all_spike_rates = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            if is_corridor_task and has_phase:
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
                    monitor.reset()
                    out_t = net(frame[:, t])
                    sr = monitor.avg_rate()
                    all_spike_rates.append(sr)
                    total_spikes += monitor.total_spikes()
                    total_frames += B

                    for i in range(B):
                        all_preds.append(out_t[i].item())
                        all_labels.append(label[i, t].item())
                        if meta is not None:
                            try:
                                phases = meta.get('phase', None) \
                                    if isinstance(meta, dict) else None
                                ph = phases[i] if phases else ''
                            except (IndexError, TypeError):
                                ph = ''
                            all_phases.append(str(ph))

                    if (encoding == 'rate'
                            and hasattr(net, 'backbone')):
                        functional.reset_net(net.backbone)
            else:
                monitor.reset()
                out_fr = net(frame)
                sr = monitor.avg_rate()
                all_spike_rates.append(sr)
                total_spikes += monitor.total_spikes()

                B = label.shape[0]
                total_frames += B

                for i in range(B):
                    all_preds.append(out_fr[i].item())
                    all_labels.append(label[i].item())
                    if meta is not None:
                        try:
                            phases = meta.get('phase', None) \
                                if isinstance(meta, dict) else None
                            ph = phases[i] if phases else ''
                        except (IndexError, TypeError):
                            ph = ''
                        all_phases.append(str(ph))

                functional.reset_net(net)

    monitor.remove()

    # ---- [4/6] 计算指标 ----
    print('\n[4/6] 计算指标...')
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    residuals = preds_arr - labels_arr

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    max_abs_error = float(np.max(np.abs(residuals)))
    mean_abs_pred = float(np.mean(np.abs(preds_arr)))
    mean_abs_gt = float(np.mean(np.abs(labels_arr)))
    avg_sr = float(np.mean(all_spike_rates)) if all_spike_rates else 0
    sparsity = 1.0 - avg_sr
    spk_per_img = total_spikes / max(total_frames, 1)
    grp_rates = {}

    # 最后一个 batch 的 group_rates (代表性)
    # 重新跑一个 batch 获取分层发放率
    net.eval()
    monitor2 = SpikeMonitor()
    monitor2.register(net)
    with torch.no_grad():
        for batch in loader:
            frame = batch[0].float().to(args.device)
            monitor2.reset()
            if is_sequence:
                if hasattr(net, 'reset_state'):
                    net.reset_state()
                out = net(frame[:, 0])
            else:
                out = net(frame)
            grp_rates = monitor2.group_rates()
            functional.reset_net(net)
            break
    monitor2.remove()

    # Zero Baseline: 预测全零
    zero_preds = np.zeros_like(labels_arr)
    zero_baseline_mae = float(np.mean(np.abs(zero_preds - labels_arr)))
    zero_baseline_rmse = float(np.sqrt(np.mean((zero_preds - labels_arr) ** 2)))

    print(f'  MAE:            {mae:.6f}')
    print(f'  RMSE:           {rmse:.6f}')
    print(f'  Max |Error|:    {max_abs_error:.6f}')
    print(f'  Mean |Pred|:    {mean_abs_pred:.6f}')
    print(f'  Mean |GT|:      {mean_abs_gt:.6f}')
    print(f'  Spike Rate:     {avg_sr:.6f}')
    print(f'  Sparsity:       {sparsity:.2%}')
    print(f'  Spikes/Image:   {spk_per_img:.0f}')
    print(f'  Total Frames:   {total_frames}')

    print(f'\n  ── Zero Baseline 对比 ──')
    print(f'  模型 MAE:       {mae:.6f}   |  Zero MAE:  {zero_baseline_mae:.6f}')
    print(f'  模型 RMSE:      {rmse:.6f}   |  Zero RMSE: {zero_baseline_rmse:.6f}')
    mae_improve = ((zero_baseline_mae - mae) / zero_baseline_mae * 100
                   if zero_baseline_mae > 0 else 0)
    rmse_improve = ((zero_baseline_rmse - rmse) / zero_baseline_rmse * 100
                    if zero_baseline_rmse > 0 else 0)
    print(f'  MAE 改善:       {mae_improve:+.1f}%')
    print(f'  RMSE 改善:      {rmse_improve:+.1f}%')
    if mae_improve < 15:
        print(f'  ⚠ 警告: 模型可能退化为近零输出 '
              f'(MAE 仅优于 zero baseline {mae_improve:.1f}%)')

    # Phase 统计
    phase_stats = {}
    if all_phases:
        phase_errs = defaultdict(list)
        for p, r in zip(all_phases, residuals):
            if p:
                phase_errs[p].append(r)

        for ph, errs in phase_errs.items():
            errs_arr = np.array(errs)
            phase_stats[ph] = {
                'mae': round(float(np.mean(np.abs(errs_arr))), 6),
                'rmse': round(
                    float(np.sqrt(np.mean(errs_arr ** 2))), 6),
                'count': len(errs_arr),
            }

        if phase_stats:
            print(f'\n  Phase 统计:')
            for ph, st in phase_stats.items():
                print(f'    {ph:12s}  '
                      f'MAE={st["mae"]:.4f}  '
                      f'RMSE={st["rmse"]:.4f}  '
                      f'n={st["count"]}')

    # ---- [5/6] 图表输出 ----
    print('\n[5/6] 生成图表...')

    # 5.1 Prediction vs GT
    plot_prediction_vs_gt(
        all_preds, all_labels, args.out_dir, exp_name, mae, rmse,
        zero_mae=zero_baseline_mae, zero_rmse=zero_baseline_rmse)

    # 5.2 Residual Histogram
    plot_residual_hist(residuals.tolist(), args.out_dir, exp_name)

    # 5.3 Phase Metrics (若有)
    if phase_stats:
        plot_phase_metrics(phase_stats, args.out_dir, exp_name)
    else:
        print('  [!] 无 phase 信息，跳过 phase_metrics.png')

    # 5.4 Spike Analysis
    plot_spike_analysis_reg(
        grp_rates, spk_per_img, mae, rmse, args.out_dir, exp_name)

    # 5.5 Training Curves (TensorBoard)
    if args.tb_dir and os.path.isdir(args.tb_dir):
        plot_training_curves_reg(args.tb_dir, args.out_dir)
    else:
        print('  [!] 未指定 tb_dir 或目录不存在，跳过训练曲线')

    # ---- [6/6] 保存文件 ----
    print('\n[6/6] 保存结果...')

    # metrics.json
    metrics = {
        'exp_name': exp_name,
        'dataset': dataset_type,
        'mode': mode,
        'neuron_type': neuron_type,
        'residual_mode': residual_mode,
        'T': T,
        'encoding': encoding,
        'img_h': img_h,
        'img_w': img_w,
        'epoch': epoch,
        'test_mae': round(mae, 6),
        'test_rmse': round(rmse, 6),
        'zero_baseline_mae': round(zero_baseline_mae, 6),
        'zero_baseline_rmse': round(zero_baseline_rmse, 6),
        'max_abs_error': round(max_abs_error, 6),
        'mean_abs_pred': round(mean_abs_pred, 6),
        'mean_abs_gt': round(mean_abs_gt, 6),
        'avg_spike_rate': round(avg_sr, 6),
        'sparsity': round(sparsity, 6),
        'spikes_per_image': round(spk_per_img, 1),
        'total_test_frames': total_frames,
        'group_rates': {k: round(v, 6) for k, v in grp_rates.items()},
    }
    if phase_stats:
        metrics['phase_stats'] = phase_stats
    if seq_len:
        metrics['seq_len'] = seq_len
        metrics['stride'] = stride

    metrics_path = os.path.join(args.out_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'  [✓] {metrics_path}')

    # predictions.csv
    csv_path = os.path.join(args.out_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if all_phases:
            writer.writerow(['index', 'prediction', 'ground_truth',
                             'residual', 'phase'])
            for i in range(len(all_preds)):
                writer.writerow([
                    i, round(all_preds[i], 6),
                    round(all_labels[i], 6),
                    round(float(residuals[i]), 6),
                    all_phases[i] if i < len(all_phases) else '',
                ])
        else:
            writer.writerow(['index', 'prediction', 'ground_truth',
                             'residual'])
            for i in range(len(all_preds)):
                writer.writerow([
                    i, round(all_preds[i], 6),
                    round(all_labels[i], 6),
                    round(float(residuals[i]), 6),
                ])
    print(f'  [✓] {csv_path}')

    # ---- 总结 ----
    print(f'\n{"=" * 70}')
    print(f'  回归评估完成  [{exp_name}]')
    print(f'{"=" * 70}')
    print(f'  MAE:    {mae:.4f}')
    print(f'  RMSE:   {rmse:.4f}')
    print(f'  SR:     {avg_sr:.4f}')
    print(f'  Frames: {total_frames}')
    if phase_stats:
        for ph, st in phase_stats.items():
            print(f'  {ph}: MAE={st["mae"]:.4f} RMSE={st["rmse"]:.4f}')
    print(f'\n  输出目录: {os.path.abspath(args.out_dir)}')
    print('=' * 70)


if __name__ == '__main__':
    main()
