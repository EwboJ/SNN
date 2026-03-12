# 进展：derive_straight_keep_dataset.py

## 完成时间

2026-03-12 14:45

## 完成内容

### 新建 `scripts/derive_straight_keep_dataset.py`

从长直行纠偏 runs 派生 `straight_keep_reg_v1` 回归数据集。

**核心流程：**
1. 缓冲裁剪：去掉开头/结尾各 500ms 的启停缓冲段
2. 阶段检测：
   - Correcting: |angular_z| 明显非零的纠偏阶段
   - Settled: 末尾基本回正、角速度低于动态阈值的稳定阶段
   - 动态阈值 = clamp(median(|w|) × 0.3, 0.02, 0.15)
3. Settled 帧数上限：默认最多保留 20 帧 Settled
4. 输出兼容原始 corridor 格式（images/, labels.csv, meta.json）
5. labels.csv 新增 phase, t_rel_ms, run_name, split 字段
6. 生成 dataset_summary.json + skipped_runs.csv

**用法：**
```bash
python scripts/derive_straight_keep_dataset.py \
    --src_root ./data/straight_keep_raw \
    --dst_root ./data/straight_keep/straight_keep_reg_v1 \
    --force
```
