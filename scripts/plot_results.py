"""
走廊导航实验可视化脚本
=====================================
从训练好的模型生成：
  1. 混淆矩阵 (confusion_matrix.png + .csv)
  2. 训练 Loss/Acc 曲线 (从 TensorBoard events 提取)
  3. Spike Rate 分析图 (分层柱状图)
  4. 每类精确率/召回率/F1 表格

用法:
    python scripts/plot_results.py \
        --ckpt checkpoint/corridor/corridor_discrete_3cls_rate_APLIF_ADD_T4/best_model.ckpt \
        --data_root ./data/corridor/test \
        --tb_dir checkpoint/corridor/corridor_discrete_3cls_rate_APLIF_ADD_T4/runs \
        --out_dir results/baseline_APLIF_ADD_T4

    # corridor_task stage3 示例（3类: Approach/Turn/Recover）
    python scripts/plot_results.py \
        --ckpt checkpoint/corridor_task/corridor_task_stage3_APLIF_ADD_T4/best_model.ckpt \
        --data_root ./data/stage1/stage3_v1/test \
        --tb_dir checkpoint/corridor_task/corridor_task_stage3_APLIF_ADD_T4/runs \
        --out_dir results/stage3_APLIF_ADD_T4
"""

import os
import sys
import json
import csv
import time
import argparse
from collections import OrderedDict, Counter

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import BaseNode
import torch.utils.data as data
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------- 尝试导入可选依赖 ----------
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

# ---------- 全局样式 ----------
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 中文支持 fallback
for font in ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']:
    try:
        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_TASK_CLASS_NAMES = {
    'action3_balanced': ['Left', 'Straight', 'Right'],
    'junction_lr': ['Left', 'Right'],
    'stage3': ['Approach', 'Turn', 'Recover'],
    'stage4': ['Follow', 'Approach', 'Turn', 'Recover'],
}


# ============================================================================
# SpikeMonitor
# ============================================================================
class SpikeMonitor:
    def __init__(self):
        self.layer_spikes = {}
        self.layer_elements = {}
        self.layer_sample_spikes = {}
        self.layer_sample_elements = {}
        self.current_batch_size = None
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

                out_detached = out.detach()
                if out_detached.ndim == 0:
                    sample_spikes = out_detached.reshape(1).to(
                        device='cpu', dtype=torch.float64)
                    sample_elements = torch.ones(
                        1, dtype=torch.float64, device='cpu')
                else:
                    bsz = self.current_batch_size
                    batch_axis = 0
                    if bsz is not None and bsz > 0:
                        if out_detached.shape[0] == bsz:
                            batch_axis = 0
                        elif out_detached.ndim > 1 and out_detached.shape[1] == bsz:
                            batch_axis = 1
                        else:
                            for ax_i, ax_n in enumerate(out_detached.shape):
                                if ax_n == bsz:
                                    batch_axis = ax_i
                                    break
                    moved = out_detached.movedim(batch_axis, 0)
                    flat = moved.reshape(moved.shape[0], -1)
                    sample_spikes = flat.sum(dim=1).to(
                        device='cpu', dtype=torch.float64)
                    sample_elements = torch.full(
                        (flat.shape[0],), float(flat.shape[1]),
                        dtype=torch.float64, device='cpu')

                self.layer_sample_spikes[name] = sample_spikes
                self.layer_sample_elements[name] = sample_elements
        return fn

    def reset(self):
        self.layer_spikes.clear()
        self.layer_elements.clear()
        self.layer_sample_spikes.clear()
        self.layer_sample_elements.clear()

    def avg_rate(self):
        s = sum(self.layer_spikes.values())
        e = sum(self.layer_elements.values())
        return s / e if e > 0 else 0

    def total_spikes(self):
        return sum(self.layer_spikes.values())

    def set_batch_size(self, batch_size):
        """设置当前 forward 的 batch 大小，用于样本级 spike 统计。"""
        self.current_batch_size = int(batch_size) if batch_size is not None else None

    def sample_rates_counts(self, expected_batch_size=None):
        """
        返回当前 forward 的样本级指标:
          - sample_rates: 每个样本的 spike_rate
          - sample_counts: 每个样本的 spike_count
        """
        if not self.layer_sample_spikes:
            return [], []

        names = sorted(self.layer_sample_spikes.keys())
        first = self.layer_sample_spikes.get(names[0])
        if not isinstance(first, torch.Tensor):
            return [], []

        bsz = expected_batch_size
        if bsz is None:
            bsz = self.current_batch_size
        if bsz is None:
            bsz = int(first.numel())
        bsz = int(max(0, bsz))
        if bsz <= 0:
            return [], []

        sample_spikes = torch.zeros(bsz, dtype=torch.float64, device='cpu')
        sample_elements = torch.zeros(bsz, dtype=torch.float64, device='cpu')
        for name in names:
            spikes = self.layer_sample_spikes.get(name)
            elems = self.layer_sample_elements.get(name)
            if not isinstance(spikes, torch.Tensor) or not isinstance(elems, torch.Tensor):
                continue
            n = min(bsz, int(spikes.numel()), int(elems.numel()))
            if n <= 0:
                continue
            sample_spikes[:n] += spikes.reshape(-1)[:n]
            sample_elements[:n] += elems.reshape(-1)[:n]

        sample_rates = torch.where(
            sample_elements > 0,
            sample_spikes / sample_elements,
            torch.zeros_like(sample_spikes))
        return sample_rates.tolist(), sample_spikes.tolist()

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
                key = parts[0] if parts[0].startswith('layer') else 'other'
            if key not in groups:
                groups[key] = [0.0, 0]
            groups[key][0] += self.layer_spikes[name]
            groups[key][1] += self.layer_elements[name]
        return {k: v[0]/v[1] if v[1] > 0 else 0 for k, v in groups.items()}

    def remove(self):
        for h in self.handles:
            h.remove()


# ============================================================================
# 1. 混淆矩阵
# ============================================================================
def plot_confusion_matrix(cm, names, out_dir, accuracy):
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 计数
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (Count)', fontsize=13)
    for i in range(len(names)):
        for j in range(len(names)):
            color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 右: 百分比
    ax2 = axes[1]
    im2 = ax2.imshow(cm_norm, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(names)))
    ax2.set_yticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=11)
    ax2.set_yticklabels(names, fontsize=11)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_title(f'Confusion Matrix (%) — Acc={accuracy:.2%}', fontsize=13)
    for i in range(len(names)):
        for j in range(len(names)):
            pct = cm_norm[i,j] * 100
            color = 'white' if pct > 50 else 'black'
            ax2.text(j, i, f'{pct:.1f}%', ha='center', va='center',
                     fontsize=12, fontweight='bold', color=color)
    fig.colorbar(im2, ax=ax2, fraction=0.046)

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


# ============================================================================
# 2. Loss / Acc 曲线 (从 TensorBoard)
# ============================================================================
def plot_training_curves(tb_dir, out_dir):
    if not HAS_TB:
        print('  [!] tensorboard 未安装，跳过训练曲线')
        print('      安装: pip install tensorboard')
        return False

    ea = EventAccumulator(tb_dir)
    ea.Reload()
    available = ea.Tags().get('scalars', [])

    if not available:
        print(f'  [!] 未找到 TensorBoard 标量数据: {tb_dir}')
        return False

    print(f'  可用指标: {available}')

    # --- Loss 曲线 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax = axes[0]
    # 兼容 v1 (TestLoss) 和 v2 (ValLoss) 标签名
    for tags, color, label in [
        (['Epoch/TrainLoss'], '#2196F3', 'Train Loss'),
        (['Epoch/ValLoss', 'Epoch/TestLoss'], '#FF5722', 'Val Loss'),
    ]:
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                ax.plot(steps, vals, color=color, label=label, linewidth=1.5)
                break
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training / Val Loss', fontsize=13)
    ax.legend(fontsize=11)

    # Accuracy or MAE
    ax2 = axes[1]
    acc_plotted = False
    # 兼容 v1 (TestAcc) 和 v2 (ValAcc)
    for tags, color, label in [
        (['Epoch/TrainAcc'], '#4CAF50', 'Train Acc'),
        (['Epoch/ValAcc', 'Epoch/TestAcc'], '#9C27B0', 'Val Acc'),
    ]:
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                ax2.plot(steps, vals, color=color, label=label, linewidth=1.5)
                acc_plotted = True
                break

    if not acc_plotted:
        for tags, color, label in [
            (['Epoch/TrainMAE'], '#4CAF50', 'Train MAE'),
            (['Epoch/ValMAE', 'Epoch/TestMAE'], '#9C27B0', 'Val MAE'),
        ]:
            for tag in tags:
                if tag in available:
                    events = ea.Scalars(tag)
                    steps = [e.step for e in events]
                    vals = [e.value for e in events]
                    ax2.plot(steps, vals, color=color, label=label, linewidth=1.5)
                    break

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy' if acc_plotted else 'MAE', fontsize=12)
    ax2.set_title('Training / Val Accuracy' if acc_plotted
                  else 'Training / Val MAE', fontsize=13)
    if acc_plotted:
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [✓] {path}')

    # --- Spike Rate 曲线 ---
    fig2, ax3 = plt.subplots(figsize=(8, 4))
    for tags, color, label in [
        (['Epoch/TrainSpikeRate'], '#FF9800', 'Train Spike Rate'),
        (['Epoch/ValSpikeRate', 'Epoch/TestSpikeRate'], '#E91E63', 'Val Spike Rate'),
    ]:
        for tag in tags:
            if tag in available:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                ax3.plot(steps, vals, color=color, label=label, linewidth=1.5)
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
# 3. Spike 分层分析图
# ============================================================================
def plot_spike_analysis(group_rates, spikes_per_img, accuracy, out_dir, exp_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: 分层发放率柱状图
    ax = axes[0]
    groups = list(group_rates.keys())
    rates = [group_rates[g] for g in groups]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    bars = ax.bar(groups, rates, color=colors[:len(groups)], edgecolor='white',
                  linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{rate:.4f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_ylabel('Spike Rate', fontsize=12)
    ax.set_title('Layer-wise Spike Rate', fontsize=13)
    ax.axhline(y=np.mean(rates), color='red', linestyle='--', alpha=0.5,
               label=f'Avg={np.mean(rates):.4f}')
    ax.legend(fontsize=10)

    # 右: 关键指标卡片
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


# ============================================================================
# 4. Per-class 指标表
# ============================================================================
def plot_per_class_metrics(per_class, out_dir):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    rows = []
    for info in per_class:
        rows.append([
            info['name'],
            f"{info['precision']:.4f}",
            f"{info['recall']:.4f}",
            f"{info['f1']:.4f}",
            str(info['support']),
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # 样式
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


# ============================================================================
# 5. 逐样本预测导出
# ============================================================================
def export_predictions_csv(test_ds, all_preds, all_labels, all_probs,
                           sample_spike_rates, sample_spike_counts,
                           class_names, num_actions, out_dir):
    """
    导出逐样本预测结果到 predictions.csv 和 errors_only.csv。
    """
    optional_fields = ['junction_id', 'turn_dir', 'stage', 'segment_id']
    has_fields = []
    samples = test_ds.samples if hasattr(test_ds, 'samples') else None
    if samples and len(samples) > 0:
        for f in optional_fields:
            if f in samples[0]:
                has_fields.append(f)

    # 构建表头
    header = ['sample_id', 'run_name', 'frame_idx',
              'true_label', 'pred_label', 'correct']
    header += [f'prob_{i}' for i in range(num_actions)]
    # 兼容保留 spike_rate（历史字段），并新增样本级细粒度字段
    header += ['spike_rate', 'sample_spike_rate', 'sample_spike_count', 'split']
    header += has_fields

    rows = []
    for i in range(len(all_labels)):
        sample = samples[i] if samples and i < len(samples) else {}

        # run_name
        run_name = sample.get('run_name', '')
        if not run_name and 'run_dir' in sample:
            run_name = os.path.basename(sample['run_dir'])
        # run_name 兜底：避免后续 run-level 统计因空值失败
        if not run_name:
            run_name = f'unknown_run_{i}'

        # frame_idx: 从图片文件名提取
        frame_idx = ''
        img_path = sample.get('img_path', '')
        if img_path:
            frame_idx = os.path.splitext(os.path.basename(img_path))[0]

        true_label = all_labels[i]
        pred_label = all_preds[i]
        correct = 1 if true_label == pred_label else 0

        row = [i, run_name, frame_idx, true_label, pred_label, correct]
        # 概率
        for c in range(num_actions):
            row.append(f'{all_probs[i][c]:.6f}')
        # spike_rate
        sr = sample_spike_rates[i] if i < len(sample_spike_rates) else 0.0
        row.append(f'{sr:.6f}')
        # sample_spike_rate / sample_spike_count
        row.append(f'{sr:.6f}')
        sc = sample_spike_counts[i] if i < len(sample_spike_counts) else 0.0
        row.append(f'{float(sc):.6f}')
        # split
        row.append('test')
        # 可选 metadata
        for f in has_fields:
            row.append(sample.get(f, ''))

        rows.append(row)

    # predictions.csv
    pred_path = os.path.join(out_dir, 'predictions.csv')
    with open(pred_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'  [✓] {pred_path} ({len(rows)} samples)')

    # errors_only.csv
    error_rows = [r for r in rows if r[5] == 0]
    err_path = os.path.join(out_dir, 'errors_only.csv')
    with open(err_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(error_rows)
    print(f'  [✓] {err_path} ({len(error_rows)} errors)')


# ============================================================================
# 主流程
# ============================================================================
def _safe_int(v, default=None):
    """安全转 int，失败返回 default。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def normalize_task_name(task_name):
    """
    Normalize task_name for robust template lookup.
    """
    if task_name is None:
        return ''
    try:
        name = str(task_name)
    except Exception:
        return ''
    name = name.strip()
    if not name:
        return ''

    normalized = name.lower().replace('-', '_').strip()
    if normalized.startswith('stage3'):
        return 'stage3'
    if normalized.startswith('stage4'):
        return 'stage4'
    if normalized.startswith('junction_lr'):
        return 'junction_lr'
    if normalized.startswith('action3_balanced'):
        return 'action3_balanced'
    return normalized


def parse_cli_class_names(cli_class_names):
    """
    Parse comma-separated class names from CLI/config input.
    Returns list[str] or None.
    """
    if cli_class_names is None:
        return None

    if isinstance(cli_class_names, str):
        text = cli_class_names.strip()
        if not text:
            return None
        names = [part.strip() for part in text.split(',')]
    elif isinstance(cli_class_names, (list, tuple)):
        names = [str(part).strip() for part in cli_class_names]
    else:
        text = str(cli_class_names).strip()
        if not text:
            return None
        names = [part.strip() for part in text.split(',')]

    names = [n for n in names if n]
    return names if names else None


def _generate_generic_class_names(num_actions):
    """Fallback generic class names by class count."""
    n = _safe_int(num_actions, None)
    if n is None or n <= 0:
        return []
    if n == 2:
        return ['class_0', 'class_1']
    if n == 3:
        return ['class_0', 'class_1', 'class_2']
    if n == 4:
        return ['class_0', 'class_1', 'class_2', 'class_3']
    return [f'class_{i}' for i in range(n)]


def _ensure_class_names_for_count(class_names, num_actions, warn_prefix=''):
    """
    Ensure class_names length matches the expected class count.
    If mismatch, rebuild generic names to avoid crashes.
    """
    n = _safe_int(num_actions, None)
    if n is None or n <= 0:
        return parse_cli_class_names(class_names) or []

    names = parse_cli_class_names(class_names) or []
    if len(names) != n:
        prefix = f'{warn_prefix}: ' if warn_prefix else ''
        print(f'  [!] {prefix}class_names length={len(names)} != '
              f'num_actions={n}, fallback to generic names')
        return _generate_generic_class_names(n)
    return names


def resolve_class_names(task_name, num_actions, cfg=None, data_root=None,
                        cli_class_names=None):
    """
    Resolve class names with robust compatibility.
    Priority:
      1) config.class_names
      2) --class_names
      3) normalized task mapping
      4) generic class_i fallback
    """
    cfg = cfg if isinstance(cfg, dict) else {}

    n = _safe_int(num_actions, None)
    if n is None or n <= 0:
        n = _safe_int(cfg.get('task_num_classes', cfg.get('num_classes', None)), None)
    if n is None or n <= 0:
        n = 3

    cfg_class_names = parse_cli_class_names(cfg.get('class_names', None))
    if cfg_class_names:
        if len(cfg_class_names) == n:
            return cfg_class_names
        print(f'  [!] config.class_names length={len(cfg_class_names)} '
              f'!= num_actions={n}, ignore config.class_names')

    cli_names = parse_cli_class_names(cli_class_names)
    if cli_names:
        if len(cli_names) == n:
            return cli_names
        print(f'  [!] --class_names length={len(cli_names)} '
              f'!= num_actions={n}, ignore --class_names')

    normalized = normalize_task_name(task_name)
    if not normalized:
        inferred = _infer_task_name_from_path(data_root, default_name='')
        normalized = normalize_task_name(inferred)
    mapped = DEFAULT_TASK_CLASS_NAMES.get(normalized, None)
    if mapped:
        if len(mapped) == n:
            return list(mapped)
        print(f'  [!] default mapping for "{normalized}" has {len(mapped)} classes '
              f'!= num_actions={n}, fallback to generic names')

    return _generate_generic_class_names(n)


def _majority_vote_int(values):
    """
    多数投票（整数标签）。
    若出现并列，取标签 id 更小者，保证结果稳定可复现。
    """
    ints = []
    for v in values:
        vi = _safe_int(v, None)
        if vi is not None:
            ints.append(vi)
    if not ints:
        return None
    cnt = Counter(ints)
    return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _infer_task_name_from_path(data_root, default_name='action3_balanced'):
    """当 ckpt 缺 task_name 时，尝试从 data_root 推断任务名。"""
    if not data_root:
        return normalize_task_name(default_name)
    path_l = str(data_root).lower().replace('-', '_').strip()
    for n in ('stage3', 'stage4', 'junction_lr', 'action3_balanced'):
        if n in path_l:
            return n
    return normalize_task_name(default_name)


def _infer_missing_task_name(data_root, num_actions, default_name=''):
    """
    Conservative fallback used only when raw task_name is missing.
    Inference is based on data_root + num_actions, and normalized.
    """
    inferred_from_path = _infer_task_name_from_path(data_root, default_name='')
    if inferred_from_path:
        return normalize_task_name(inferred_from_path)

    cls_n = _safe_int(num_actions, None)
    if cls_n == 2:
        return 'junction_lr'
    if cls_n == 4:
        return 'stage4'
    if cls_n == 3:
        return 'action3_balanced'
    return normalize_task_name(default_name)


def export_run_level_summary(out_dir, class_names=None):
    """
    基于 predictions.csv 的 run_name 做逐 run 多数投票统计：
      - run_predictions.csv
      - run_metrics.json
    Returns:
      dict or None
    """
    pred_path = os.path.join(out_dir, 'predictions.csv')
    if not os.path.isfile(pred_path):
        print('  [!] 未找到 predictions.csv，跳过 run-level 统计')
        return None

    run_bucket = OrderedDict()
    with open(pred_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for ridx, row in enumerate(reader):
            raw_name = (row.get('run_name') or '').strip()
            # run_name 缺失时安全兜底，避免不同样本被错误合并
            if raw_name:
                run_name = raw_name
            else:
                sid = row.get('sample_id', ridx)
                run_name = f'unknown_run_{sid}'

            if run_name not in run_bucket:
                run_bucket[run_name] = {'true': [], 'pred': []}
            run_bucket[run_name]['true'].append(row.get('true_label'))
            run_bucket[run_name]['pred'].append(row.get('pred_label'))

    def _label_name(label_id):
        if label_id is None:
            return ''
        if class_names and 0 <= label_id < len(class_names):
            return class_names[label_id]
        return f'class_{label_id}'

    out_rows = []
    correct_runs = 0
    for run_name, d in run_bucket.items():
        tmaj = _majority_vote_int(d['true'])
        pmaj = _majority_vote_int(d['pred'])
        ok = int((tmaj is not None) and (pmaj is not None) and (tmaj == pmaj))
        correct_runs += ok
        out_rows.append({
            'run_name': run_name,
            'true_label_majority': '' if tmaj is None else tmaj,
            'pred_label_majority': '' if pmaj is None else pmaj,
            'correct': ok,
            'num_frames': len(d['true']),
            # 额外字段：方便论文表格直接读标签名
            'true_name_majority': _label_name(tmaj),
            'pred_name_majority': _label_name(pmaj),
        })

    run_pred_path = os.path.join(out_dir, 'run_predictions.csv')
    with open(run_pred_path, 'w', newline='', encoding='utf-8') as f:
        fields = [
            'run_name',
            'true_label_majority',
            'pred_label_majority',
            'correct',
            'num_frames',
            'true_name_majority',
            'pred_name_majority',
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f'  [✓] {run_pred_path} ({len(out_rows)} runs)')

    total_runs = len(out_rows)
    run_acc = (correct_runs / total_runs) if total_runs > 0 else 0.0
    run_metrics = {
        'run_accuracy': round(float(run_acc), 6),
        'total_runs': total_runs,
        'correct_runs': correct_runs,
        'source': 'predictions.csv_majority_vote',
    }
    run_met_path = os.path.join(out_dir, 'run_metrics.json')
    with open(run_met_path, 'w', encoding='utf-8') as f:
        json.dump(run_metrics, f, indent=2, ensure_ascii=False)
    print(f'  [✓] {run_met_path}')
    return run_metrics


def main():
    parser = argparse.ArgumentParser(
        description='走廊导航实验可视化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='测试数据路径 (包含 run 目录)')
    parser.add_argument('--tb_dir', type=str, default=None,
                        help='TensorBoard events 目录 (可选)')
    parser.add_argument('--out_dir', type=str, default='./results/baseline',
                        help='输出目录')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_names', type=str, default=None,
                        help='Comma-separated class names, e.g. '
                             'Approach,Turn,Recover')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 70)
    print('  走廊导航实验可视化')
    print('=' * 70)

    # ---- 加载模型 ----
    print('\n[1/5] 加载模型...')
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}

    # 判断数据集类型
    dataset_type = cfg.get('dataset', 'corridor')  # corridor / corridor_task / cifar10
    is_corridor_task = (dataset_type == 'corridor_task')

    encoding = cfg.get('encoding', 'rate')
    T = cfg.get('T', 4)
    neuron_type = cfg.get('neuron_type', 'APLIF')
    residual_mode = cfg.get('residual_mode', 'ADD')
    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    task_name_raw = cfg.get('task_name', '')
    if task_name_raw is None:
        task_name_raw = ''
    elif not isinstance(task_name_raw, str):
        task_name_raw = str(task_name_raw)
    task_name_raw = task_name_raw.strip()
    task_name_normalized = normalize_task_name(task_name_raw)
    task_num_classes = cfg.get('task_num_classes', cfg.get('num_classes', None))

    # corridor_task 已知任务 -> 类名映射表
    if is_corridor_task:
        # corridor_task: 从 config 读取 task_name / task_num_classes
        num_actions = _safe_int(
            cfg.get('task_num_classes', cfg.get('num_classes', None)), None)
        if num_actions is None or num_actions <= 0:
            num_actions = _safe_int(task_num_classes, None)
        if num_actions is None or num_actions <= 0:
            num_actions = 3
        task_num_classes = num_actions
        if not task_name_raw:
            task_name_raw = _infer_missing_task_name(
                args.data_root, num_actions, default_name='')
        mode = 'discrete'  # corridor_task 都是离散分类
        control_dim = 1
        action_set = str(num_actions)  # 兼容字段
        task_name_normalized = normalize_task_name(task_name_raw)
        exp_task_name = task_name_raw if task_name_raw else (
            task_name_normalized if task_name_normalized else 'unknown_task')
        exp_name = f"{exp_task_name}_{neuron_type}_{residual_mode}_T{T}"
        # 优先按 task_name 查表，未知任务则自动生成
        class_names = resolve_class_names(
            task_name=task_name_raw,
            num_actions=num_actions,
            cfg=cfg,
            data_root=args.data_root,
            cli_class_names=args.class_names)
        class_names = _ensure_class_names_for_count(
            class_names, num_actions, warn_prefix='corridor_task class_names')
    else:
        mode = cfg.get('mode', 'discrete')
        action_set = str(cfg.get('action_set', '3'))
        control_dim = cfg.get('control_dim', 1)
        num_actions = cfg.get('num_classes', 3 if action_set == '3' else 5)
        task_num_classes = task_num_classes if task_num_classes is not None else num_actions
        exp_name = f"{neuron_type}_{residual_mode}_T{T}"
        names_3 = ['Left', 'Straight', 'Right']
        names_5 = ['Forward', 'Backward', 'Left', 'Right', 'Stop']
        class_names = names_3 if action_set == '3' else names_5
        class_names = _ensure_class_names_for_count(
            class_names, num_actions, warn_prefix='default class_names')

    is_discrete = (mode == 'discrete')

    print(f'  Config: {exp_name}, dataset={dataset_type}, mode={mode}, '
          f'encoding={encoding}, epoch={epoch}')
    if is_corridor_task:
        task_name_normalized = normalize_task_name(task_name_raw)
        print(f'  dataset_type : {dataset_type}')
        print(f'  task_name_raw        : {task_name_raw}')
        print(f'  task_name_normalized : {task_name_normalized}')
        print(f'  num_actions  : {num_actions}')
        print(f'  class_names  : {class_names}')

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from models.snn_corridor import build_corridor_net

    net = build_corridor_net(
        head_type=mode, num_actions=num_actions, control_dim=control_dim,
        encoding=encoding, T=T, neuron_type=neuron_type,
        residual_mode=residual_mode)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        net.load_state_dict(ckpt['model_state_dict'])
    else:
        net.load_state_dict(ckpt)
    net.to(args.device)
    net.eval()
    print(f'  [✓] 模型加载成功')

    # ---- 数据集 ----
    print('\n[2/5] 加载测试数据...')
    img_h = cfg.get('img_h', 32)
    img_w = cfg.get('img_w', 32)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_h, img_w)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    if is_corridor_task:
        from datasets.corridor_task_dataset import CorridorTaskDataset
        test_ds = CorridorTaskDataset(
            root_dir=args.data_root, transforms=transform,
            print_stats=True)
    else:
        from datasets.corridor_dataset import CorridorDataset
        test_ds = CorridorDataset(
            root_dir=args.data_root, mode=mode, control_dim=control_dim,
            valid_only=True, action_set=action_set, transforms=transform)

    loader = data.DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    print(f'  [✓] 测试集: {len(test_ds)} 帧')
    print(f'  [✓] 类名: {class_names}')

    # ---- 推理 ----
    print('\n[3/5] 推理...')
    monitor = SpikeMonitor()
    monitor.register(net)

    all_preds, all_labels = [], []
    all_probs = []
    sample_spike_rates = []
    sample_spike_counts = []
    total_spikes, total_frames = 0, 0
    all_spike_rates = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Inference'):
            monitor.reset()
            images = images.float().to(args.device)
            monitor.set_batch_size(images.shape[0])
            out = net(images)

            if is_discrete:
                preds = out.argmax(1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())
                probs = F.softmax(out, dim=1).cpu().numpy()
                all_probs.append(probs)

            batch_sr = monitor.avg_rate()
            batch_total_spikes = monitor.total_spikes()
            total_spikes += batch_total_spikes
            total_frames += images.shape[0]
            all_spike_rates.append(batch_sr)
            sample_rates_batch, sample_counts_batch = monitor.sample_rates_counts(
                expected_batch_size=images.shape[0])
            if (len(sample_rates_batch) != images.shape[0]
                    or len(sample_counts_batch) != images.shape[0]):
                sample_rates_batch = [batch_sr] * images.shape[0]
                sample_counts_batch = [batch_total_spikes / max(images.shape[0], 1)] * images.shape[0]
            sample_spike_rates.extend([float(v) for v in sample_rates_batch])
            sample_spike_counts.extend([float(v) for v in sample_counts_batch])
            functional.reset_net(net)

    if is_discrete and all_probs:
        all_probs = np.concatenate(all_probs, axis=0)

    group_rates = monitor.group_rates()
    monitor.remove()

    # 计算指标
    accuracy = 0.0
    run_metrics = None
    macro_precision, macro_recall, macro_f1 = None, None, None
    if is_discrete:
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_labels)

    spikes_per_img = total_spikes / max(total_frames, 1)
    avg_sr = np.mean(all_spike_rates)

    print(f'  [✓] 准确率: {accuracy:.4f}, SpikeRate: {avg_sr:.4f}, '
          f'Spikes/Img: {spikes_per_img:.0f}')

    # ---- 生成图表 ----
    print('\n[4/5] 生成图表...')

    # 4a. 混淆矩阵
    if is_discrete:
        class_names = _ensure_class_names_for_count(
            class_names, num_actions, warn_prefix='confusion_matrix class_names')
        cm = np.zeros((num_actions, num_actions), dtype=int)
        for p, l in zip(all_preds, all_labels):
            cm[l][p] += 1
        plot_confusion_matrix(cm, class_names, args.out_dir,
                              accuracy)

        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f1, sup = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(num_actions)),
            zero_division=0)
        per_class = []
        for i in range(num_actions):
            per_class.append({
                'name': class_names[i],
                'precision': float(p[i]),
                'recall': float(r[i]),
                'f1': float(f1[i]),
                'support': int(sup[i]),
            })
        # 宏平均指标：更适合论文中跨类整体分析
        macro_precision = float(np.mean(p)) if len(p) > 0 else 0.0
        macro_recall = float(np.mean(r)) if len(r) > 0 else 0.0
        macro_f1 = float(np.mean(f1)) if len(f1) > 0 else 0.0
        plot_per_class_metrics(per_class, args.out_dir)

        # 4d. 逐样本预测导出
        export_predictions_csv(
            test_ds, all_preds, all_labels, all_probs,
            sample_spike_rates, sample_spike_counts,
            class_names, num_actions, args.out_dir)
        run_metrics = export_run_level_summary(args.out_dir, class_names)
        print(f'  [i] Frame-level Accuracy: {accuracy:.4f}')
        if run_metrics:
            print(f'  [i] Run-level   Accuracy: '
                  f'{run_metrics["run_accuracy"]:.4f} '
                  f'({run_metrics["correct_runs"]}/{run_metrics["total_runs"]})')
        else:
            print('  [i] Run-level   Accuracy: N/A')

    # 4b. Spike 分析
    plot_spike_analysis(group_rates, spikes_per_img, accuracy,
                        args.out_dir, exp_name)

    # 4c. 训练曲线 (从 TensorBoard)
    tb_dir = args.tb_dir
    if not tb_dir:
        # 自动查找
        ckpt_dir = os.path.dirname(args.ckpt)
        runs_dir = os.path.join(ckpt_dir, 'runs')
        if os.path.isdir(runs_dir):
            tb_dir = runs_dir
        elif os.path.isdir(ckpt_dir):
            # events 可能直接在 ckpt 目录
            events = [f for f in os.listdir(ckpt_dir)
                      if f.startswith('events.out.tfevents')]
            if events:
                tb_dir = ckpt_dir

    if tb_dir:
        plot_training_curves(tb_dir, args.out_dir)
    else:
        print('  [!] 未找到 TensorBoard 目录，跳过训练曲线')

    # 代理功耗细粒度统计（按 true class / 正误分组）
    class_spike_stats = OrderedDict()
    correct_vs_error_spike_stats = {
        'correct': {'mean_spike_rate': 0.0, 'mean_spike_count': 0.0, 'count': 0},
        'error': {'mean_spike_rate': 0.0, 'mean_spike_count': 0.0, 'count': 0},
    }
    if is_discrete and len(all_labels) > 0:
        n = len(all_labels)

        def _sample_sr(i):
            return float(sample_spike_rates[i]) if i < len(sample_spike_rates) else 0.0

        def _sample_sc(i):
            return float(sample_spike_counts[i]) if i < len(sample_spike_counts) else 0.0

        def _agg_indices(indices):
            if not indices:
                return {'mean_spike_rate': 0.0, 'mean_spike_count': 0.0, 'count': 0}
            rates = np.array([_sample_sr(i) for i in indices], dtype=np.float64)
            counts = np.array([_sample_sc(i) for i in indices], dtype=np.float64)
            return {
                'mean_spike_rate': round(float(np.mean(rates)), 6),
                'mean_spike_count': round(float(np.mean(counts)), 6),
                'count': int(len(indices)),
            }

        for cls_id in range(num_actions):
            cls_indices = [i for i in range(n) if int(all_labels[i]) == cls_id]
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
            st = _agg_indices(cls_indices)
            st['class_id'] = int(cls_id)
            class_spike_stats[cls_name] = st

        correct_indices = [i for i in range(n) if int(all_preds[i]) == int(all_labels[i])]
        error_indices = [i for i in range(n) if int(all_preds[i]) != int(all_labels[i])]
        correct_vs_error_spike_stats = {
            'correct': _agg_indices(correct_indices),
            'error': _agg_indices(error_indices),
        }

    # ---- 保存 metrics.json ----
    print('\n[5/5] 保存 metrics.json...')
    # 元信息：优先从 ckpt config 读取，缺失时做安全 fallback
    dataset_meta = cfg.get(
        'dataset',
        'corridor_task' if is_corridor_task else 'corridor')
    if dataset_meta not in ('corridor_task', 'corridor'):
        dataset_meta = 'corridor_task' if is_corridor_task else 'corridor'

    task_name_raw_meta = cfg.get('task_name', None)
    if task_name_raw_meta is None:
        task_name_raw_meta = ''
    elif not isinstance(task_name_raw_meta, str):
        task_name_raw_meta = str(task_name_raw_meta)
    task_name_raw_meta = task_name_raw_meta.strip()
    # Only infer when task_name is missing.
    if is_corridor_task and not task_name_raw_meta:
        infer_num_actions = _safe_int(
            cfg.get('task_num_classes', cfg.get('num_classes', num_actions)),
            num_actions)
        task_name_raw_meta = _infer_missing_task_name(
            args.data_root, infer_num_actions, default_name='')

    task_num_classes_meta = cfg.get('task_num_classes', task_num_classes)
    if task_num_classes_meta is None:
        task_num_classes_meta = cfg.get('num_classes', num_actions)
    task_name_normalized_meta = normalize_task_name(task_name_raw_meta)
    if is_corridor_task and not task_name_normalized_meta:
        infer_num_actions = _safe_int(task_num_classes_meta, num_actions)
        task_name_normalized_meta = normalize_task_name(
            _infer_missing_task_name(
                args.data_root, infer_num_actions, default_name=''))

    seq_len_meta = cfg.get('seq_len', None)
    stride_meta = cfg.get('stride', None)
    encoding_meta = cfg.get('encoding', encoding)

    metrics = {
        'exp_name': exp_name,
        'epoch': epoch,
        'dataset': dataset_meta,
        'task_name': task_name_raw_meta,
        'task_name_raw': task_name_raw_meta,
        'task_name_normalized': task_name_normalized_meta,
        'normalized_task_name': task_name_normalized_meta,
        'task_num_classes': _safe_int(task_num_classes_meta, task_num_classes_meta),
        'neuron_type': cfg.get('neuron_type', neuron_type),
        'residual_mode': cfg.get('residual_mode', residual_mode),
        'T': _safe_int(cfg.get('T', T), T),
        'img_h': _safe_int(cfg.get('img_h', img_h), img_h),
        'img_w': _safe_int(cfg.get('img_w', img_w), img_w),
        'seq_len': seq_len_meta,
        'stride': stride_meta,
        'encoding': encoding_meta,
        'accuracy': round(accuracy, 6),
        'avg_spike_rate': round(float(avg_sr), 6),
        'sparsity': round(1.0 - float(avg_sr), 6),
        'spikes_per_image': round(spikes_per_img, 1),
        'total_spikes': round(float(total_spikes), 1),
        'test_samples': total_frames,
        'total_test_frames': total_frames,
        'group_rates': {k: round(v, 6) for k, v in group_rates.items()},
        'class_spike_stats': class_spike_stats if is_discrete else {},
        'correct_vs_error_spike_stats': (
            correct_vs_error_spike_stats if is_discrete else {}),
        'source': 'plot_results.py',
    }
    if is_discrete:
        metrics['per_class'] = per_class
        metrics['confusion_matrix'] = cm.tolist()
        metrics['macro_precision'] = round(float(macro_precision or 0.0), 6)
        metrics['macro_recall'] = round(float(macro_recall or 0.0), 6)
        metrics['macro_f1'] = round(float(macro_f1 or 0.0), 6)
        if run_metrics:
            metrics['run_accuracy'] = run_metrics['run_accuracy']
            metrics['run_total'] = run_metrics['total_runs']
            metrics['run_correct'] = run_metrics['correct_runs']

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'  [✓] metrics.json')

    # ---- 汇总 ----
    print('\n' + '=' * 70)
    print(f'  所有输出已保存到: {os.path.abspath(args.out_dir)}/')
    print('=' * 70)
    for fn in sorted(os.listdir(args.out_dir)):
        fp = os.path.join(args.out_dir, fn)
        sz = os.path.getsize(fp) / 1024
        print(f'    {fn:30s}  {sz:6.1f} KB')
    print('=' * 70)


if __name__ == '__main__':
    main()
