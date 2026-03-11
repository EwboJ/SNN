# train.py (v2) 与 train copy.py (v1) 功能对比

## 1. 概述

两个脚本均为 SNN 论文消融实验框架的统一训练入口，支持 CIFAR-10/100 和走廊导航数据集。  
`train copy.py` 是 **v1 原始版本**（971 行），`train.py` 是 **v2 重构版本**（1121 行）。

---

## 2. 共同功能

| 功能 | 说明 |
|------|------|
| **数据集支持** | CIFAR-10/100 图像分类 + 走廊导航（离散分类 / 连续回归 / 序列） |
| **SNN 神经元** | LIF / PLIF / ALIF / APLIF，残差模式 standard / ADD |
| **Spike 监控** | SpikeMonitor 类，支持可反传 spike rate 正则化、分层发放率统计 |
| **损失函数** | CrossEntropy / MSE-OneHot / SmoothL1(Huber)，自动选择 |
| **类别平衡** | weighted_sampler / class_weight |
| **训练功能** | AMP 混合精度、Resume 断点恢复、TensorBoard 日志 |
| **优化器** | Adam / SGD，StepLR / CosineAnnealing 调度 |
| **Checkpoint** | 定期保存 + 最佳模型保存 |

---

## 3. 关键差异

### 3.1 数据划分策略

| | v1 (`train copy.py`) | v2 (`train.py`) |
|---|---|---|
| **划分方式** | train / test 二分 | train / val / test 三分 |
| **最佳模型选择依据** | test 集指标 | val 集指标 |
| **test 集用途** | 每个 epoch 都评估，兼做验证集 | 仅在 `--final_test` 时最终评估 |
| **val 缺失处理** | 无 val 概念 | 自动回退到 test 替代 val |

> v2 遵循 ML 最佳实践：训练时仅看 val，避免 test 信息泄漏。

### 3.2 评估方式

| | v1 | v2 |
|---|---|---|
| **实现方式** | 内联在训练循环中（代码重复） | 抽取为独立函数 `run_evaluation()` |
| **返回值** | 分散变量 | 统一 dict：`loss, metric, avg_sr, avg_sp, avg_spk_img, samples` |
| **复用性** | 无 | val / test 共用同一函数 |

### 3.3 新增命令行参数（v2 独有）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--img_h` | 32 | 走廊图像输入高度（可改 64 等） |
| `--img_w` | 32 | 走廊图像输入宽度（可改 96 等） |
| `--corridor_hflip` | 关闭 | 是否启用水平翻转增强（默认关闭，因 Left/Right 标签不会自动互换） |
| `--final_test` | 关闭 | 训练结束后用 best_model 对 test 集做最终评估并输出 JSON |

v1 共 **39** 个参数，v2 共 **43** 个参数。

### 3.4 图像尺寸处理

| | v1 | v2 |
|---|---|---|
| **走廊图像 Resize** | 硬编码 32×32 | 参数化 `--img_h` × `--img_w` |
| **水平翻转** | 训练时始终开启 | 默认关闭，需 `--corridor_hflip` 显式开启 |

### 3.5 TensorBoard 标量命名

| 指标 | v1 | v2 |
|------|-----|-----|
| 验证/测试损失 | `Epoch/TestLoss` | `Epoch/ValLoss` |
| 验证/测试准确率 | `Epoch/TestAcc` | `Epoch/ValAcc` |
| 验证/测试 Spike Rate | `Epoch/TestSpikeRate` | `Epoch/ValSpikeRate` |
| 验证/测试 Sparsity | `Epoch/TestSparsity` | `Epoch/ValSparsity` |

> v2 将 epoch 循环中的 "Test" 一律改为 "Val"，语义更准确。

### 3.6 Config 快照

| | v1 | v2 |
|---|---|---|
| **内容** | 基础参数（neuron_type, T, seed 等） | 基础 + `img_h, img_w, corridor_hflip` |
| **JSON 导出** | 无 | `--final_test` 时生成 `final_test_metrics.json` |
| **json 导入** | 无 | `import json` |

### 3.7 `build_corridor_dataset()` 返回值

| | v1 | v2 |
|---|---|---|
| **返回元组** | 6 项：`train_ds, test_ds, num_out, 3, is_sequence, sampler` | 7 项：`train_ds, val_ds, test_ds, num_out, 3, is_sequence, sampler` |

---

## 4. v2 相比 v1 的改进总结

1. **train/val/test 三分** — 避免在 test 上做模型选择导致过拟合
2. **评估函数抽取** — 消除代码重复，提高可维护性
3. **图像尺寸可配置** — 支持非正方形输入（如 64×96）
4. **水平翻转默认关闭** — 避免走廊场景 Left/Right 标签语义错误
5. **最终测试流程** — `--final_test` 一键完成 best_model 加载 + test 评估 + JSON 输出
6. **TensorBoard 语义修正** — 区分 Val 和 Test 标量

---

## 5. 使用建议

- **日常训练开发**：使用 `train.py`（v2），享受完整三分验证和灵活图像尺寸
- **复现旧实验**：如需对比旧结果，可临时使用 `train copy.py`（v1）
- `train copy.py` 可作为历史备份保留，后续新实验统一使用 v2
