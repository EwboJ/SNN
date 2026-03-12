# 进展：verify_task_datasets.py

## 完成时间

2026-03-12 15:25

## 完成内容

### 新建 `scripts/verify_task_datasets.py`

统一任务数据集核查可视化脚本，替代/扩展 `verify_stage1_windows.py`。

**支持任务类型：**
- junction_lr_v1 — 路口二分类
- stage4_v1 — 四阶段分类
- action3_balanced_v1 — 三分类
- straight_keep_reg_v1 — 直行纠偏回归
- loop_sparse/* — Loop 窗口/稀疏直行

**输出：**
- preview_*.png — strip 缩略图 + 标签条 + phase 条
- timeline_*.png — angular_z 时间轴 (phase 背景色)
- verify_summary.json — 全局统计 + per-run 统计 + 异常警告

**用法：**
```bash
# 自动检测
python scripts/verify_task_datasets.py --data_root ./data/stage1/junction_lr_v1

# straight_keep 回归
python scripts/verify_task_datasets.py --data_root ./data/straight_keep/straight_keep_reg_v1

# loop_sparse
python scripts/verify_task_datasets.py --data_root ./data/loop_sparse/junction_windows
```
