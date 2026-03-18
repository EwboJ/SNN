# 进展：新建 plot_regression_results.py

## 完成时间

2026-03-18 14:45

## 新增文件

### `scripts/plot_regression_results.py`

专门服务 corridor regression / straight_keep_reg 任务的评估和可视化。

**输入参数：**
| 参数 | 说明 |
|------|------|
| `--ckpt` | 模型 checkpoint 路径 |
| `--data_root` | 测试数据路径 |
| `--tb_dir` | TensorBoard events 目录 |
| `--out_dir` | 输出目录 |
| `--device` | 设备 |
| `--batch_size` | 批大小 |

**从 checkpoint config 读取：** dataset, mode, encoding, T, neuron_type, residual_mode, img_h/w, seq_len, stride

**图表输出：**
1. `prediction_vs_gt.png` — 预测 vs 真值散点 + 时间序列对比
2. `residual_hist.png` — 残差直方图 + Q-Q plot
3. `phase_metrics.png` — Correcting/Settled 分段指标对比（若有 phase）
4. `spike_analysis.png` — SNN 分层发放率柱状图 + 指标卡片
5. `training_curves.png` — Loss/MAE 训练曲线 + Spike Rate 曲线

**数值输出：**
- `metrics.json` — 完整指标（MAE, RMSE, max_abs_error, spike 统计, phase_stats 等）
- `predictions.csv` — 逐样本预测结果（含 phase 列）

**指标：** MAE, RMSE, max_abs_error, mean_abs_pred/gt, avg_spike_rate, sparsity, spikes_per_image, group_rates, phase MAE/RMSE

**兼容性：**
- 若传入离散分类模型，直接提示并退出
- 若无 phase 元数据，跳过 phase 相关输出

## 用法

```bash
python scripts/plot_regression_results.py \
    --ckpt checkpoint/corridor/straight_keep_reg_APLIF_ADD_T4/best_model.ckpt \
    --data_root ./data/straight_keep/straight_keep_reg_v1/test \
    --tb_dir checkpoint/corridor/straight_keep_reg_APLIF_ADD_T4/runs \
    --out_dir results/straight_keep_reg_APLIF_ADD_T4
```
