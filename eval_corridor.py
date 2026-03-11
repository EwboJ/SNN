"""
走廊导航离线评估脚本
================================
对训练好的 CorridorPolicyNet 做可复现实验统计。

离散模式:
    python eval_corridor.py -weights best_model.ckpt --data_root ./data/corridor/test
    → accuracy, per-class P/R/F1, confusion matrix (png+csv)

回归模式:
    python eval_corridor.py -weights best_model.ckpt --data_root ./data/corridor/test
    → MAE/RMSE, 预测 vs GT 曲线 (png)

输出目录: results/<exp_name>/
    metrics.json, config.json, confusion_matrix.csv/.png, curves.png
"""

import os
import sys
import json
import csv
import time
import argparse
from collections import OrderedDict

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
import seaborn as sns

plt.rc('font', family='Times New Roman')

_seed_ = 42


# ============================================================================
# SpikeMonitor (评测用，不需要反传)
# ============================================================================
class SpikeMonitor:
    def __init__(self):
        self.spike_counts = {}
        self.handles = []

    def register(self, net):
        for name, module in net.named_modules():
            if isinstance(module, BaseNode):
                self.handles.append(
                    module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(mod, inp, out):
            if isinstance(out, torch.Tensor):
                self.spike_counts[name] = (out.sum().item(), out.numel())
        return fn

    def reset(self):
        self.spike_counts.clear()

    def avg_rate(self):
        if not self.spike_counts:
            return 0.0
        s = sum(v[0] for v in self.spike_counts.values())
        e = sum(v[1] for v in self.spike_counts.values())
        return s / e if e > 0 else 0.0

    def total_spikes(self):
        return sum(v[0] for v in self.spike_counts.values())

    def layer_rates(self):
        groups = {}
        for name, (spk, elem) in self.spike_counts.items():
            key = name.split('.')[0]
            if not key.startswith('layer'):
                key = 'stem'
            if key not in groups:
                groups[key] = [0.0, 0]
            groups[key][0] += spk
            groups[key][1] += elem
        return {k: v[0]/v[1] if v[1] > 0 else 0 for k, v in groups.items()}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ============================================================================
# 评估核心
# ============================================================================
def evaluate(args):
    # 种子
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # ---- 加载权重 & config ----
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'config' in ckpt:
        cfg = ckpt['config']
    else:
        cfg = {}

    mode = cfg.get('mode', args.mode)
    encoding = cfg.get('encoding', args.encoding)
    action_set = cfg.get('action_set', args.action_set)
    control_dim = cfg.get('control_dim', args.control_dim)
    T = cfg.get('T', args.T)
    neuron_type = cfg.get('neuron_type', args.neuron_type)
    residual_mode = cfg.get('residual_mode', args.residual_mode)
    exp_name = cfg.get('exp_name', f'corridor_{mode}')
    num_actions = 3 if action_set == '3' else 5

    # 命令行覆盖
    if args.mode != 'discrete':
        mode = args.mode
    if args.encoding != 'rate':
        encoding = args.encoding

    is_discrete = (mode == 'discrete')

    print(f'[Config] mode={mode}, encoding={encoding}, T={T}, '
          f'neuron={neuron_type}, residual={residual_mode}')

    # ---- 构建网络 ----
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.snn_corridor import build_corridor_net

    net = build_corridor_net(
        head_type=mode,
        num_actions=num_actions,
        control_dim=control_dim,
        encoding=encoding,
        T=T,
        neuron_type=neuron_type,
        residual_mode=residual_mode,
    )

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        net.load_state_dict(ckpt['model_state_dict'])
        print(f'[√] Loaded epoch {ckpt.get("epoch", "?")}')
    else:
        net.load_state_dict(ckpt)

    net.to(args.device)
    net.eval()

    monitor = SpikeMonitor()
    monitor.register(net)

    # ---- 数据集 ----
    from datasets.corridor_dataset import CorridorDataset

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    test_ds = CorridorDataset(
        root_dir=args.data_root,
        mode=mode,
        control_dim=control_dim,
        valid_only=True,
        action_set=action_set,
        transforms=transform,
    )
    loader = data.DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, drop_last=False,
                             num_workers=args.j)

    # ---- 推理 ----
    all_preds = []
    all_labels = []
    all_pred_vals = []   # regression: raw predictions
    all_label_vals = []  # regression: raw targets
    total_spikes = 0
    total_frames = 0
    spike_rates = []
    layer_rates_accum = {}
    forward_times = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f'Eval [{exp_name}]'):
            monitor.reset()
            images = images.float().to(args.device)
            labels = labels.to(args.device)

            # 计时
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            out = net(images)

            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            bs = images.shape[0]
            forward_times.append((t1 - t0) / bs)  # per-image time

            sr = monitor.avg_rate()
            spike_rates.append(sr)
            total_spikes += monitor.total_spikes()
            total_frames += bs

            for lname, lrate in monitor.layer_rates().items():
                if lname not in layer_rates_accum:
                    layer_rates_accum[lname] = []
                layer_rates_accum[lname].append(lrate)

            if is_discrete:
                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
            else:
                all_pred_vals.extend(out.cpu().numpy().tolist())
                all_label_vals.extend(labels.cpu().numpy().tolist())

            functional.reset_net(net)

    # ---- 计算指标 ----
    metrics = OrderedDict()
    metrics['exp_name'] = exp_name
    metrics['mode'] = mode
    metrics['total_frames'] = total_frames

    # SNN 指标
    avg_sr = float(np.mean(spike_rates))
    spk_per_frame = total_spikes / max(total_frames, 1)
    avg_fwd_ms = float(np.mean(forward_times)) * 1000
    fps = 1000.0 / avg_fwd_ms if avg_fwd_ms > 0 else 0

    metrics['avg_spike_rate'] = round(avg_sr, 6)
    metrics['sparsity'] = round(1.0 - avg_sr, 6)
    metrics['spikes_per_frame'] = round(spk_per_frame, 1)
    metrics['avg_forward_ms'] = round(avg_fwd_ms, 3)
    metrics['fps'] = round(fps, 1)
    metrics['layer_rates'] = {k: round(float(np.mean(v)), 6)
                              for k, v in sorted(layer_rates_accum.items())}

    # ---- 输出目录 ----
    out_dir = os.path.join(args.out_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    if is_discrete:
        _eval_discrete(all_preds, all_labels, num_actions, action_set,
                       metrics, out_dir)
    else:
        _eval_regression(all_pred_vals, all_label_vals, control_dim,
                         metrics, out_dir)

    # ---- 保存 ----
    # metrics.json
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # config.json (可复现)
    config_out = {
        'weights': args.weights,
        'data_root': args.data_root,
        'seed': args.seed,
        'device': args.device,
        'mode': mode,
        'encoding': encoding,
        'T': T,
        'neuron_type': neuron_type,
        'residual_mode': residual_mode,
        'action_set': action_set,
        'control_dim': control_dim,
        'checkpoint_config': cfg,
    }
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config_out, f, indent=2, ensure_ascii=False)

    # ---- 打印报告 ----
    print('\n' + '=' * 70)
    print(f'  评估报告  [{exp_name}]')
    print('=' * 70)
    for k, v in metrics.items():
        if k == 'layer_rates':
            print(f'  分层发放率:')
            for lk, lv in v.items():
                print(f'    {lk:10s}: {lv:.4f}')
        elif k == 'per_class':
            print(f'  Per-class 指标:')
            for cls_info in v:
                print(f"    {cls_info['name']:10s}  "
                      f"P={cls_info['precision']:.3f}  "
                      f"R={cls_info['recall']:.3f}  "
                      f"F1={cls_info['f1']:.3f}  "
                      f"(n={cls_info['support']})")
        elif k == 'confusion_matrix':
            continue
        else:
            print(f'  {k:20s}: {v}')
    print('=' * 70)
    print(f'  结果已保存: {out_dir}/')
    print('=' * 70)


# ============================================================================
# 离散评估
# ============================================================================
def _eval_discrete(preds, labels, num_actions, action_set, metrics, out_dir):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)

    if action_set == '3':
        names = ['Left', 'Straight', 'Right']
    else:
        names = ['Forward', 'Backward', 'Left', 'Right', 'Stop']

    acc = accuracy_score(labels, preds)
    p, r, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_actions)), zero_division=0)

    metrics['accuracy'] = round(float(acc), 6)
    metrics['per_class'] = []
    for i in range(num_actions):
        metrics['per_class'].append({
            'id': i,
            'name': names[i] if i < len(names) else f'class_{i}',
            'precision': round(float(p[i]), 4),
            'recall': round(float(r[i]), 4),
            'f1': round(float(f1[i]), 4),
            'support': int(sup[i]),
        })

    cm = confusion_matrix(labels, preds, labels=list(range(num_actions)))
    metrics['confusion_matrix'] = cm.tolist()

    # 保存 CSV
    csv_path = os.path.join(out_dir, 'confusion_matrix.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = [''] + names[:num_actions]
        writer.writerow(header)
        for i in range(num_actions):
            row = [names[i] if i < len(names) else f'class_{i}']
            row.extend(cm[i].tolist())
            writer.writerow(row)

    # 保存 PNG
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=names[:num_actions],
                yticklabels=names[:num_actions],
                annot_kws={'size': 14}, ax=ax)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title(f'Confusion Matrix (Acc={acc:.4f})', fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# 回归评估
# ============================================================================
def _eval_regression(pred_vals, label_vals, control_dim, metrics, out_dir):
    preds = np.array(pred_vals)
    labels = np.array(label_vals)

    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    dim_names = ['angular_z', 'linear_x'] if control_dim == 1 \
        else ['linear_x', 'angular_z']
    if control_dim == 1:
        dim_names = ['angular_z']

    overall_mae = float(np.mean(np.abs(preds - labels)))
    overall_rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    metrics['mae'] = round(overall_mae, 6)
    metrics['rmse'] = round(overall_rmse, 6)

    metrics['per_dim'] = {}
    for d in range(min(control_dim, preds.shape[1])):
        name = dim_names[d] if d < len(dim_names) else f'dim_{d}'
        mae_d = float(np.mean(np.abs(preds[:, d] - labels[:, d])))
        rmse_d = float(np.sqrt(np.mean((preds[:, d] - labels[:, d]) ** 2)))
        metrics['per_dim'][name] = {
            'mae': round(mae_d, 6),
            'rmse': round(rmse_d, 6),
        }

    # 绘制预测 vs GT 曲线
    n = len(preds)
    fig, axes = plt.subplots(control_dim, 1,
                             figsize=(14, 4 * control_dim), dpi=150)
    if control_dim == 1:
        axes = [axes]

    x_axis = np.arange(n)
    for d in range(control_dim):
        ax = axes[d]
        name = dim_names[d] if d < len(dim_names) else f'dim_{d}'
        mae_d = metrics['per_dim'][name]['mae']

        ax.plot(x_axis, labels[:, d], label='GT', color='#2196F3',
                linewidth=0.8, alpha=0.8)
        ax.plot(x_axis, preds[:, d], label='Pred', color='#FF5722',
                linewidth=0.8, alpha=0.8)
        ax.fill_between(x_axis,
                        labels[:, d] - np.abs(preds[:, d] - labels[:, d]),
                        labels[:, d] + np.abs(preds[:, d] - labels[:, d]),
                        alpha=0.1, color='#FF5722')
        ax.set_ylabel(name, fontsize=13)
        ax.set_title(f'{name}  (MAE={mae_d:.4f})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frame Index', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 散点图: pred vs GT
    fig2, axes2 = plt.subplots(1, control_dim,
                               figsize=(6 * control_dim, 5), dpi=150)
    if control_dim == 1:
        axes2 = [axes2]

    for d in range(control_dim):
        ax = axes2[d]
        name = dim_names[d] if d < len(dim_names) else f'dim_{d}'
        ax.scatter(labels[:, d], preds[:, d], s=3, alpha=0.4,
                   color='#673AB7')
        lims = [min(labels[:, d].min(), preds[:, d].min()),
                max(labels[:, d].max(), preds[:, d].max())]
        ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='y=x')
        ax.set_xlabel(f'GT {name}', fontsize=12)
        ax.set_ylabel(f'Pred {name}', fontsize=12)
        ax.set_title(f'{name}: Pred vs GT', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'scatter.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='走廊导航离线评估',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-weights', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='测试数据根目录 (含 images/ + labels.csv)')
    parser.add_argument('--out_dir', type=str, default='./results',
                        help='结果输出目录')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('-j', type=int, default=4, help='worker 数')

    # 手动覆盖 (当 checkpoint 无 config 时)
    parser.add_argument('--mode', default='discrete',
                        choices=['discrete', 'regression'])
    parser.add_argument('--encoding', default='rate',
                        choices=['rate', 'framediff'])
    parser.add_argument('-T', default=8, type=int)
    parser.add_argument('--neuron_type', default='APLIF')
    parser.add_argument('--residual_mode', default='ADD')
    parser.add_argument('--action_set', default='3', choices=['3', '5'])
    parser.add_argument('--control_dim', default=1, type=int, choices=[1, 2])

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
