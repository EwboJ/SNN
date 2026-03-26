# 进展：verify_task_datasets.py — 支持 stage3_v1

## 完成时间

2026-03-26 14:45

## 修改文件

### `scripts/verify_task_datasets.py`

新增 stage3_v1 数据集核查支持。

**修改点：**

1. **KNOWN_TASKS** 新增 `'stage3'` 条目（classification, has_phase=True）

2. **Docstring** 更新：
   - 列表新增 stage3_v1 (三阶段: Approach/Turn/Recover)
   - 新增 stage3 用法示例命令

3. **Timeline 图** t_turn_on 标记条件新增 `'stage3'`

4. **Type Label** 新增 `'stage3': '三阶段分类 (Approach/Turn/Recover)'`

5. **核查提示** 新增 stage3 专用：
   - phase 条: Approach→Turn→Recover 应无 Follow
   - timeline: Turn 阶段应有明显 angular_z 变化
   - label 条: 三类颜色与 phase 一致

6. **verify_summary.json** 新增 `per_split` 节：
   - 每个 split 的 runs 数、frames 数、label_distribution

**颜色方案：**
- PHASE_COLORS / LABEL_COLORS 已有 Approach/Turn/Recover，无需修改
