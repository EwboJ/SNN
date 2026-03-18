# 进展：增强 regression final_test_metrics

## 完成时间

2026-03-18 14:35

## 修改文件

### `train.py`

**新增函数：**

| 函数 | 说明 |
|------|------|
| `_run_regression_final_test(net, loader, is_sequence, args)` | 在 final_test 时对 regression 做增强评估：计算全局 RMSE，按 phase 分段统计 MAE/RMSE |
| `_extract_phase(meta, sample_idx, seq_t=None)` | 从 DataLoader 返回的 meta 中提取 phase 字符串，兼容单帧/序列/collated dict 格式 |

**实现策略：**
- 临时将 dataset 的 `return_meta` 设为 True 获取 phase 元数据
- 评估完毕后恢复原始 `return_meta` 值
- 逐样本收集 error，按 phase 分桶计算统计
- 优先展示 Correcting / Settled 两个阶段

**final_test_metrics.json 新增字段（仅 regression）：**
- `test_rmse` — 全局 RMSE
- `phase_available` — 是否有 phase 信息
- `phase_stats` — 按 phase 分段的 MAE/RMSE/count

**终端输出示例：**
```
  [Regression 增强评估]
  Test RMSE:     0.0523
  Phase 统计:
    Correcting    MAE=0.0821  RMSE=0.0934  n=1250
    Settled       MAE=0.0234  RMSE=0.0287  n=890
```

**兼容性：**
- 分类任务（is_discrete=True）完全不受影响
- 无 phase 元数据时 `phase_available=false`，不报错
- 原有 `run_evaluation()` 和训练主循环完全不变
