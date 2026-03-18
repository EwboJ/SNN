# 进展：新建 collect_experiment_results.py

## 完成时间

2026-03-18 10:35

## 新增文件

### `scripts/collect_experiment_results.py`

自动汇总多个实验目录的训练/测试结果，生成 CSV 和 Markdown 消融实验表格。

**文件读取优先级：**
1. `final_test_metrics.json` — 最终独立测试结果（含 config）
2. `metrics.json` — 训练过程中保存的指标
3. `best_model.ckpt` — checkpoint 元信息（epoch、config）
4. `pipeline_log.json` — 数据管线日志（补充 task_type）

**自动提取的字段（从 config 中）：**
- neuron_type, residual_mode, T
- img_h, img_w, seq_len, stride
- dataset, task_name

**输出格式：**
- CSV: 全部字段（SUMMARY_FIELDS + 动态发现的字段）
- Markdown: 精简列，分类和回归分别使用不同列集合
  - 分类: exp_name, neuron, residual, T, val_acc, test_acc, loss, spike_rate, sparsity
  - 回归: exp_name, neuron, residual, T, val_mae, test_mae, loss, spike_rate, sparsity

**排序：**
- 分类任务: 默认按 `test_acc` 降序
- 回归任务: 默认按 `test_mae` 升序
- 可通过 `--sort_by` 自定义

**其他功能：**
- `--recursive` 递归扫描子目录
- `--top_k` 打印 Top-K 实验
- 数据来源 / Neuron / Task 分布统计

## 用法

```bash
# 递归扫描 results 目录
python scripts/collect_experiment_results.py \
    --exp_root ./results --recursive

# 指定输出路径
python scripts/collect_experiment_results.py \
    --exp_root ./results \
    --out_csv ./results/ablation_summary.csv \
    --out_md ./results/ablation_summary.md \
    --recursive --top_k 10
```
