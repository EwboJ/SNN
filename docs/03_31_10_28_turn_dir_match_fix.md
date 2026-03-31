# turn_dir_match 统计逻辑修复

**时间**: 2026-03-31 10:28  
**文件**: `scripts/replay_hierarchical_system.py`

## 问题
run 结束时 RECOVER→STRAIGHTKEEP 会清空 `locked_turn_dir`，导致 `final_locked_turn_dir` 为空，`turn_dir_match` 误判为 False。

## 修复方案
- 新增 `first_locked_turn_dir`（首次锁定方向）和 `most_frequent_locked_turn_dir`（出现最多方向）
- `turn_dir_match` 优先使用 `first_locked_turn_dir`，若为空退化到 `most_frequent_locked_turn_dir`
- `final_locked_turn_dir` 保留但仅作为"结束时状态"，不参与评价

## replay_summary.json 新增字段
| 字段 | 说明 |
|------|------|
| `first_locked_turn_dir` | 首次出现的非空锁定方向 |
| `most_frequent_locked_turn_dir` | 整个 replay 中出现次数最多的方向 |
