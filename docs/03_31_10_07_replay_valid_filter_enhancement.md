# Replay 脚本 Valid 过滤与系统级分析增强

**时间**: 2026-03-31 10:07  
**文件**: `scripts/replay_hierarchical_system.py`

## 修改内容

### 1. 新增 `--valid_only` 命令行参数
- `store_true` 类型，默认 False
- labels.csv 含 valid 字段时，仅保留 `1/true/True` 的帧

### 2. `_collect_frames` 增强
- 返回 4 元组：`(frames, label_fields, original_count, skipped_count)`
- 新增 `_is_valid_flag()` 辅助函数

### 3. replay_trace.csv 新增列
| 列名 | 来源 |
|------|------|
| `valid_flag` | labels.csv → valid |
| `gt_action_name` | labels.csv → action_name |
| `gt_label_name` | labels.csv → label_name |

### 4. replay_summary.json 新增字段
| 字段 | 说明 |
|------|------|
| `used_total_steps` | 实际参与回放的帧数 |
| `original_total_steps` | 过滤前总帧数 |
| `used_valid_only` | 是否启用了 valid 过滤 |
| `skipped_invalid_steps` | 被过滤掉的帧数 |
| `turn_signal_peak_votes` | Turn 票数历史峰值 |
| `junction_lock_first_step` | junction 首次锁定的 step |
| `fallback_step_list` | 所有 fallback 发生的 step 列表 |

### 5. debug_rows 新增 fallback 分析字段
- `fallback_blocked_by_turn_signal`
- `fallback_blocked_by_junction_lock`

### 6. 保持不变的逻辑
- 三模块推理流程
- 状态机 update() 调用
- 所有现有输出文件
- 现有命令行参数
