"""
走廊 SNN 脉冲统计分析工具
================================
对走廊导航策略网络 (CorridorPolicyNet) 进行详细的脉冲行为分析，输出：
  1. 全网平均发放率 (target ≈ 0.1)
  2. 分层发放率 (stem / layer1 / layer2 / layer3 / head)
  3. 全网 Sparsity (target > 90%)
  4. 每张图 spike count (能耗 proxy)
  5. 分类准确率 + 每类准确率 (离散模式)
  6. MAE (回归模式)
  7. 混淆矩阵 (离散模式)
  8. FPS 测量
  9. 可选：导出 CSV 用于论文作图

示例:
    # 单模型分析
    python eval_corridor_spike_stats.py -weights checkpoint/corridor/best_model.ckpt \\
        --corridor_root ./data/corridor_all

    # 手动指定参数 (当 ckpt 无 config 时)
    python eval_corridor_spike_stats.py -weights best_model.ckpt \\
        --corridor_root ./data/corridor_all \\
        --neuron_type APLIF --residual_mode ADD -T 4 --action_set 3

    # 批量对比
    python eval_corridor_spike_stats.py --batch_compare \\
        checkpoint/corridor/APLIF_ADD_T4/best_model.ckpt \\
        checkpoint/corridor/LIF_ADD_T4/best_model.ckpt \\
        --corridor_root ./data/corridor_all

    # 导出 CSV
    python eval_corridor_spike_stats.py -weights best_model.ckpt \\
        --corridor_root ./data/corridor_all --export_csv results/corridor_stats.csv
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
from collections import OrderedDict, Counter
from tqdm import tqdm

from models.snn_corridor import build_corridor_net
from datasets.corridor_dataset import (
    CorridorDataset, ACTION_3CLASS, ACTION_5CLASS,
    print_action_distribution)

_seed_ = 42
torch.manual_seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


# ============================================================================
# DetailedSpikeMonitor (与 eval_spike_stats.py 一致)
# ============================================================================
class DetailedSpikeMonitor:
    """详细的脉冲监控器，收集逐层、逐样本的 spike 统计"""

    def __init__(self):
        self.handles = []
        self.layer_spikes = {}
        self.layer_elements = {}
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

    def get_group_stats(self):
        """返回分组统计 (stem, layer1, layer2, layer3, head)"""
        groups = OrderedDict()
        for name in self.neuron_names:
            if name not in self.layer_spikes:
                continue
            parts = name.split('.')
            # backbone.layer1... → layer1
            # backbone.conv1... → stem
            # head... → head
            if 'backbone' in name:
                sub = parts[1] if len(parts) > 1 else parts[0]
                key = sub if sub.startswith('layer') else 'stem'
            elif 'head' in name:
                key = 'head'
            else:
                key = parts[0] if parts[0].startswith('layer') else 'other'

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


# ============================================================================
# 混淆矩阵
# ============================================================================
def compute_confusion_matrix(all_preds, all_labels, num_classes):
    """计算混淆矩阵 [num_classes x num_classes]"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(all_preds, all_labels):
        cm[t][p] += 1
    return cm


def print_confusion_matrix(cm, action_names):
    """打印混淆矩阵"""
    nc = cm.shape[0]
    names = [action_names.get(i, f'cls{i}') for i in range(nc)]
    max_name_len = max(len(n) for n in names)

    header = ' ' * (max_name_len + 4) + '  '.join(
        f'{n:>{max_name_len}s}' for n in names)
    print(f'  {"预测 →":>{max_name_len + 4}s}')
    print(f'  {header}')
    print('  ' + '-' * len(header))

    for i in range(nc):
        row_str = '  '.join(f'{cm[i][j]:>{max_name_len}d}' for j in range(nc))
        total = cm[i].sum()
        acc = cm[i][i] / total * 100 if total > 0 else 0
        print(f'  {names[i]:>{max_name_len}s} │ {row_str}  │ {acc:5.1f}%')


# ============================================================================
# 模型评估
# ============================================================================
def evaluate_corridor_model(
    weights_path,
    corridor_root,
    device='cuda:0',
    batch_size=32,
    neuron_type=None,
    residual_mode=None,
    T=None,
    action_set=None,
    mode=None,
    encoding=None,
):
    """
    评估走廊导航模型，返回完整统计 dict。
    """
    # 加载 checkpoint
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)

    # 读取 config
    if isinstance(ckpt, dict) and 'config' in ckpt:
        cfg = ckpt['config']
        _nt = cfg.get('neuron_type', 'APLIF')
        _rm = cfg.get('residual_mode', 'ADD')
        _T = cfg.get('T', 8)
        _mode = cfg.get('mode', 'discrete')
        _enc = cfg.get('encoding', 'rate')
        _as = cfg.get('action_set', '3')
        _cd = cfg.get('control_dim', 1)
    else:
        _nt = neuron_type or 'APLIF'
        _rm = residual_mode or 'ADD'
        _T = T or 8
        _mode = mode or 'discrete'
        _enc = encoding or 'rate'
        _as = action_set or '3'
        _cd = 1

    # 命令行覆盖
    if neuron_type: _nt = neuron_type
    if residual_mode: _rm = residual_mode
    if T: _T = T
    if mode: _mode = mode
    if encoding: _enc = encoding
    if action_set: _as = action_set

    is_discrete = (_mode == 'discrete')
    num_actions = 3 if _as == '3' else 5
    action_names = ACTION_3CLASS if _as == '3' else ACTION_5CLASS

    exp_name = f"{_nt}_{_rm}_T{_T}"
    if is_discrete:
        exp_name += f"_{_as}cls"

    # 构建网络
    net = build_corridor_net(
        head_type=_mode,
        num_actions=num_actions,
        control_dim=_cd,
        encoding=_enc,
        T=_T,
        neuron_type=_nt,
        residual_mode=_rm,
    )

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

    # 构建走廊测试数据集
    img_size = 32
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    test_root = os.path.join(corridor_root, 'test')
    if not os.path.isdir(test_root):
        # 如果 corridor_root 本身就是数据目录（无 train/test 拆分）
        test_root = corridor_root

    test_dataset = CorridorDataset(
        root_dir=test_root,
        mode=_mode,
        control_dim=_cd,
        valid_only=True,
        action_set=_as,
        transforms=test_transform,
        print_stats=True,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False, num_workers=0)

    # 推理
    correct = 0
    total = 0
    total_loss = 0.0
    total_spikes = 0
    all_group_stats = {}
    per_image_spikes = []
    all_preds = []
    all_labels = []
    total_mae = 0.0

    t_start = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Eval [{exp_name}]'):
            monitor.reset()
            images = images.float().to(device)
            labels = labels.to(device)

            out = net(images)

            if is_discrete:
                loss = F.cross_entropy(out, labels)
                _, preds = torch.max(out, 1)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
            else:
                loss = F.mse_loss(out, labels)
                total_mae += (out - labels).abs().sum().item()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            batch_spikes = monitor.get_total_spike_count()
            total_spikes += batch_spikes
            per_image_spikes.append(batch_spikes / labels.size(0))

            for gname, gstat in monitor.get_group_stats().items():
                if gname not in all_group_stats:
                    all_group_stats[gname] = {'spikes': 0.0, 'elements': 0}
                all_group_stats[gname]['spikes'] += gstat['spikes']
                all_group_stats[gname]['elements'] += gstat['elements']

            functional.reset_net(net)

    t_end = time.time()

    avg_loss = total_loss / total
    fps = total / (t_end - t_start)
    avg_spikes_per_img = total_spikes / total
    overall_rate = sum(v['spikes'] for v in all_group_stats.values()) / \
                   max(sum(v['elements'] for v in all_group_stats.values()), 1)

    group_rates = OrderedDict()
    for gname in sorted(all_group_stats.keys()):
        v = all_group_stats[gname]
        group_rates[gname] = v['spikes'] / v['elements'] \
            if v['elements'] > 0 else 0.0

    # 分类指标
    if is_discrete:
        accuracy = correct / total
        cm = compute_confusion_matrix(all_preds, all_labels, num_actions)
        per_class_acc = {}
        for c in range(num_actions):
            c_total = cm[c].sum()
            per_class_acc[action_names[c]] = (
                cm[c][c] / c_total if c_total > 0 else 0.0)
    else:
        accuracy = 0.0
        cm = None
        per_class_acc = {}

    mae = total_mae / total if not is_discrete else 0.0

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
        'mode': _mode,
        'encoding': _enc,
        'action_set': _as,
        'accuracy': accuracy,
        'mae': mae,
        'loss': avg_loss,
        'avg_spike_rate': overall_rate,
        'sparsity': 1.0 - overall_rate,
        'spikes_per_image': avg_spikes_per_img,
        'fps': fps,
        'group_rates': group_rates,
        'spike_std': float(np.std(per_image_spikes)) if per_image_spikes else 0.0,
        'confusion_matrix': cm,
        'per_class_acc': per_class_acc,
        'action_names': action_names,
        'num_samples': total,
        'is_discrete': is_discrete,
    }


# ============================================================================
# 报告输出
# ============================================================================
def print_report(result):
    """打印单模型报告"""
    print('\n' + '=' * 70)
    print(f'  走廊 SNN 脉冲统计报告  [{result["exp_name"]}]')
    print('=' * 70)
    print(f'  权重文件:         {result["weights"]}')
    print(f'  训练 Epoch:       {result["epoch"]}')
    print(f'  配置:             neuron={result["neuron_type"]}, '
          f'residual={result["residual_mode"]}, T={result["T"]}')
    print(f'  模式:             {result["mode"]} | 编码: {result["encoding"]} '
          f'| 动作集: {result["action_set"]}类')
    print(f'  测试样本数:       {result["num_samples"]}')
    print('-' * 70)

    if result['is_discrete']:
        print(f'  准确率:           {result["accuracy"]:.4f} '
              f'({result["accuracy"]*100:.2f}%)')
    else:
        print(f'  MAE:              {result["mae"]:.4f}')

    print(f'  测试损失:         {result["loss"]:.4f}')
    print(f'  全网平均发放率:   {result["avg_spike_rate"]:.4f}')
    print(f'  全网 Sparsity:    {result["sparsity"]:.2%}')
    print(f'  每张图 Spikes:    {result["spikes_per_image"]:.0f} '
          f'(± {result["spike_std"]:.0f})')
    print(f'  FPS:              {result["fps"]:.1f} img/s')

    # 分层发放率
    print()
    print('  分层发放率:')
    for gname, grate in result['group_rates'].items():
        bar_len = min(int(grate * 100), 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f'    {gname:10s}: {grate:.4f}  {bar}')

    # 每类准确率 (离散)
    if result['is_discrete'] and result['per_class_acc']:
        print()
        print('  每类准确率:')
        for name, acc in result['per_class_acc'].items():
            bar_len = min(int(acc * 20), 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f'    {name:10s}: {acc:.4f} ({acc*100:5.1f}%)  {bar}')

    # 混淆矩阵
    if result['is_discrete'] and result['confusion_matrix'] is not None:
        print()
        print('  混淆矩阵 (行=真实, 列=预测):')
        print_confusion_matrix(result['confusion_matrix'],
                               result['action_names'])

    print('=' * 70)


def print_comparison_table(results):
    """打印多模型对比表"""
    is_discrete = results[0]['is_discrete']

    print('\n' + '=' * 110)
    print('  走廊模型消融对比表')
    print('=' * 110)

    if is_discrete:
        header = (f'{"Config":<25s} {"Acc":>8s} {"Loss":>8s} '
                  f'{"SpikeRate":>10s} {"Sparsity":>10s} '
                  f'{"Spk/Img":>10s} {"FPS":>8s} {"Samples":>8s}')
    else:
        header = (f'{"Config":<25s} {"MAE":>8s} {"Loss":>8s} '
                  f'{"SpikeRate":>10s} {"Sparsity":>10s} '
                  f'{"Spk/Img":>10s} {"FPS":>8s} {"Samples":>8s}')
    print(header)
    print('-' * 110)

    for r in results:
        metric = r['accuracy'] if is_discrete else r['mae']
        line = (f'{r["exp_name"]:<25s} '
                f'{metric:>8.4f} {r["loss"]:>8.4f} '
                f'{r["avg_spike_rate"]:>10.4f} {r["sparsity"]:>9.2%} '
                f'{r["spikes_per_image"]:>10.0f} {r["fps"]:>8.1f} '
                f'{r["num_samples"]:>8d}')
        print(line)

    print('=' * 110)

    # 分层对比
    print('\n' + '=' * 110)
    print('  分层发放率对比')
    print('=' * 110)
    all_groups = set()
    for r in results:
        all_groups.update(r['group_rates'].keys())
    all_groups = sorted(all_groups)

    header2 = f'{"Config":<25s}'
    for g in all_groups:
        header2 += f' {g:>10s}'
    print(header2)
    print('-' * 110)

    for r in results:
        line2 = f'{r["exp_name"]:<25s}'
        for g in all_groups:
            rate = r['group_rates'].get(g, 0.0)
            line2 += f' {rate:>10.4f}'
        print(line2)
    print('=' * 110)

    # 每类准确率对比 (仅离散)
    if is_discrete:
        print('\n' + '=' * 110)
        print('  每类准确率对比')
        print('=' * 110)
        all_classes = set()
        for r in results:
            all_classes.update(r['per_class_acc'].keys())
        all_classes = sorted(all_classes)

        header3 = f'{"Config":<25s}'
        for c in all_classes:
            header3 += f' {c:>10s}'
        print(header3)
        print('-' * 110)

        for r in results:
            line3 = f'{r["exp_name"]:<25s}'
            for c in all_classes:
                acc = r['per_class_acc'].get(c, 0.0)
                line3 += f' {acc:>9.2%}'
            print(line3)
        print('=' * 110)


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='走廊 SNN 脉冲统计分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型
  python eval_corridor_spike_stats.py -weights checkpoint/corridor/best_model.ckpt \\
      --corridor_root ./data/corridor_all

  # 批量对比
  python eval_corridor_spike_stats.py --batch_compare \\
      checkpoint/corridor/exp1/best_model.ckpt \\
      checkpoint/corridor/exp2/best_model.ckpt \\
      --corridor_root ./data/corridor_all

  # 导出 CSV
  python eval_corridor_spike_stats.py -weights best_model.ckpt \\
      --corridor_root ./data/corridor_all --export_csv results/stats.csv
""")

    parser.add_argument('-weights', type=str, default=None,
                        help='单模型 checkpoint 路径')
    parser.add_argument('--batch_compare', nargs='+', default=None,
                        help='多模型对比：提供多个 checkpoint 路径')
    parser.add_argument('--corridor_root', type=str, required=True,
                        help='走廊数据根目录 (需包含 test/ 子目录)')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=32, type=int, help='batch size')
    parser.add_argument('--export_csv', type=str, default=None,
                        help='导出 CSV 路径')

    # 手动覆盖 (当 checkpoint 无 config 时)
    parser.add_argument('--neuron_type', default=None, type=str,
                        choices=['LIF', 'PLIF', 'ALIF', 'APLIF'])
    parser.add_argument('--residual_mode', default=None, type=str,
                        choices=['standard', 'ADD'])
    parser.add_argument('-T', default=None, type=int)
    parser.add_argument('--action_set', default=None, type=str,
                        choices=['3', '5'])
    parser.add_argument('--mode', default=None, type=str,
                        choices=['discrete', 'regression'])
    parser.add_argument('--encoding', default=None, type=str,
                        choices=['rate', 'framediff'])

    args = parser.parse_args()

    if not args.weights and not args.batch_compare:
        print("请指定 -weights 或 --batch_compare 参数")
        return

    results = []

    def eval_one(wp):
        if not os.path.exists(wp):
            print(f"[!] 跳过不存在的文件: {wp}")
            return None
        return evaluate_corridor_model(
            wp,
            corridor_root=args.corridor_root,
            device=args.device,
            batch_size=args.b,
            neuron_type=args.neuron_type,
            residual_mode=args.residual_mode,
            T=args.T,
            action_set=args.action_set,
            mode=args.mode,
            encoding=args.encoding,
        )

    if args.batch_compare:
        for wp in args.batch_compare:
            r = eval_one(wp)
            if r:
                results.append(r)
                print_report(r)
        if len(results) > 1:
            print_comparison_table(results)

    elif args.weights:
        r = eval_one(args.weights)
        if r:
            results.append(r)
            print_report(r)

    # 导出 CSV
    if args.export_csv and results:
        os.makedirs(os.path.dirname(args.export_csv) or '.', exist_ok=True)
        fieldnames = ['exp_name', 'neuron_type', 'residual_mode', 'T',
                      'mode', 'encoding', 'action_set',
                      'accuracy', 'mae', 'loss',
                      'avg_spike_rate', 'sparsity',
                      'spikes_per_image', 'fps', 'epoch',
                      'num_samples', 'weights']

        # 添加分层字段
        all_groups = set()
        for r in results:
            all_groups.update(r['group_rates'].keys())
        for g in sorted(all_groups):
            fieldnames.append(f'rate_{g}')

        # 添加每类准确率字段
        all_classes = set()
        for r in results:
            all_classes.update(r['per_class_acc'].keys())
        for c in sorted(all_classes):
            fieldnames.append(f'acc_{c}')

        with open(args.export_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {}
                for k in fieldnames:
                    if k in r and not k.startswith('rate_') \
                            and not k.startswith('acc_'):
                        row[k] = r[k]
                for g in sorted(all_groups):
                    row[f'rate_{g}'] = r['group_rates'].get(g, 0.0)
                for c in sorted(all_classes):
                    row[f'acc_{c}'] = r['per_class_acc'].get(c, 0.0)
                writer.writerow(row)
        print(f'\n[OK] CSV 已导出: {args.export_csv}')


if __name__ == '__main__':
    main()
