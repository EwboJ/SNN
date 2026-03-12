# 进展：extract_loop_windows.py 重写

## 完成时间

2026-03-12 15:16

## 完成内容

### 重写 `scripts/extract_loop_windows.py`

从旧版 flat 目录结构重写为 split-based 目录结构。

**变更要点：**
- 输入: `data/loop_eval/<split>/<run_name>/` (含 train/val/test)
- 输出: `data/loop_sparse/<task_name>/<split>/<derived_run_name>/`
- 新增 `run_extract_loop_windows()` 可调用函数 (供 pipeline 调用)
- labels.csv 新增 `split` 字段
- 按 split 分别统计

**三种提取模式：**
1. `junction_windows` — 每次 turn 前后窗口，标签=转向方向
2. `stage_windows` — Follow/Approach/Turn/Recover 四阶段
3. `sparse_follow` — 长直段均匀采样子片段

**用法：**
```bash
python scripts/extract_loop_windows.py \
    --src_root ./data/loop_eval \
    --dst_root ./data/loop_sparse \
    --mode all --force
```
