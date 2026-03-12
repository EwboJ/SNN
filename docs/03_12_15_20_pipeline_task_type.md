# 进展：corridor_dataset_pipeline.py 支持 task_type

## 完成时间

2026-03-12 15:20

## 完成内容

### 修改 `scripts/corridor_dataset_pipeline.py`

新增 `--task_type` 参数，支持 4 种数据类型的不同处理流程：

| task_type | 默认流程 | 特殊阶段 |
|-----------|---------|---------|
| junction | export → downsample → split → derive | derive_stage1_datasets |
| straight_keep | export → split → derive_straight_keep | derive_straight_keep_dataset (自动跳过 downsample) |
| loop | export → split [→ extract_windows] | extract_loop_windows (需 --loop_extract_windows) |
| generic | export → downsample → split → derive | 同 junction，兼容旧行为 |

**task_type 独立默认路径：**
- junction: junction_all → junction_balanced → junction → stage1
- straight_keep: straight_keep_all → straight_keep → straight_keep_reg_v1
- loop: loop_eval_raw → loop_eval [→ loop_sparse]
- generic: corridor_all → corridor_balanced → corridor → stage1

**用法示例：**
```bash
# junction 全流程
python scripts/corridor_dataset_pipeline.py --task_type junction --mode all --bag_dir ./data/bags --force

# straight_keep (默认跳过 downsample)
python scripts/corridor_dataset_pipeline.py --task_type straight_keep --mode all --force

# loop (导出+划分+提取窗口)
python scripts/corridor_dataset_pipeline.py --task_type loop --mode all --loop_extract_windows --force

# 兼容旧用法
python scripts/corridor_dataset_pipeline.py --mode all --force
```
