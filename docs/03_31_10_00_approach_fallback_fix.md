# APPROACH Fallback 过早回退问题修复

**时间**: 2026-03-31 10:00  
**文件**: `controllers/hierarchical_state_machine.py`

## 问题描述

在 replay 中发现 APPROACH 在 Turn 信号正在形成时（Turn=4, Approach=3）过早 fallback 到 STRAIGHTKEEP，
导致系统永远进不了 TURN 状态。原因是旧逻辑中 fallback 优先级高于 TURN 进入判断，且 fallback 条件过于宽松。

## 修改内容

### 1. APPROACH 转移优先级重排
- **TURN 进入判断优先于 fallback**
- 只有确认"既没有稳定 junction 候选，也没有正在形成的 Turn 信号"时，才允许 fallback

### 2. 宽松 TURN 进入条件（三种路径）
| 条件 | 说明 |
|------|------|
| A | junction 已锁定 + Turn 票数 >= majority 阈值 |
| B | junction 已锁定 + Turn 票数 >= stage_enter_turn_votes |
| C | junction 已锁定 + Turn 票数 >= (enter_turn_votes - 1) + 最近帧 Turn 趋势上升 |

条件 C 专门用于吸收 stage3 对 Turn 类偏弱的问题。

### 3. 保守 fallback 条件（四个同时满足）
- `state_step >= stage_window_size`
- `approach_votes < majority_threshold`
- `turn_votes < max(1, enter_turn_votes - 1)`
- `junction_candidate is None`

### 4. Debug 字段增强
- `fallback_blocked_by_turn_signal: bool`
- `fallback_blocked_by_junction_lock: bool`

### 5. Transition Reason 细化
- `normal_enter_turn` — 标准进入 TURN
- `enter_turn_with_locked_junction_candidate` — 宽松条件进入 TURN
- `approach_fallback_no_turn_signal` — 确认无 Turn 信号后 fallback
- `turn_timeout` — TURN 超时

## 未修改的逻辑
- TURN 中 locked_turn_dir 不可修改
- TURN -> RECOVER 的连续 Recover 支持逻辑
- RECOVER 的 blend 恢复逻辑
- omega 裁剪逻辑
- update() 接口签名与 replay_hierarchical_system.py 完全兼容
