# TURN → RECOVER 转移逻辑增强

**时间**: 2026-03-31 10:32  
**文件**: `controllers/hierarchical_state_machine.py`, `scripts/replay_hierarchical_system.py`

## 问题
TURN → RECOVER 主要由 `turn_timeout` 触发，Recover 识别未被有效利用。旧逻辑中 Recover 不连续会立刻清零 `_recover_support_count`。

## 修改

### 状态机 (`hierarchical_state_machine.py`)
| 修改项 | 旧逻辑 | 新逻辑 |
|--------|--------|--------|
| Recover 累计 | 不连续立刻清零 | 渐减 `max(0, count-1)` |
| 转移优先级 | timeout 优先 | `recover_signal_confirmed` 优先，timeout 兜底 |
| 新参数 | 无 | `recover_support_steps_needed`（默认2） |
| reason | `consecutive_recover` / `turn_timeout` | `recover_signal_confirmed` / `turn_timeout` |

### debug 新增字段
- `current_recover_votes`、`recover_support_count`、`recover_support_threshold`

### Replay 脚本 (`replay_hierarchical_system.py`)
- `_build_state_machine` 新增 `recover_support_steps_needed` 参数传递
