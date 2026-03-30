# 层级状态机增强记录（03_30_22_24）

## 背景
本次更新目标是让 `controllers/hierarchical_state_machine.py` 与 `configs/hierarchical_nav.yaml` 的控制参数更一致，并补充系统级安全机制与可分析字段。

## 主要改动
1. `HierarchicalNavigatorStateMachine.__init__` 新增参数：
   - `max_turn_steps`
   - `use_fixed_turn_rate`
   - `omega_clip`
   - `use_clip`

2. 新增 TURN 超时保护：
   - 在 `TURN` 状态中，当 `state_step >= max_turn_steps` 时，强制转入 `RECOVER`。
   - 转移原因固定写为 `turn_timeout`，便于 replay 和论文分析复现。

3. 新增最终角速度统一裁剪：
   - 在 `omega_cmd_final` 计算完成后执行统一裁剪（可通过 `use_clip` 开关控制）。
   - 裁剪区间：`[-omega_clip, +omega_clip]`。

4. TURN 控制策略增强：
   - `use_fixed_turn_rate=True`：TURN 阶段使用固定转向角速度（保持既有行为）。
   - `use_fixed_turn_rate=False`：允许少量 `straight_keep` 混合，但锁存方向仍主导，防止反向穿越。
   - TURN 期间 `locked_turn_dir` 不可改写（保持关键约束）。

5. `debug` 新增字段：
   - `turn_timeout_triggered`
   - `clip_applied`
   - `omega_before_clip`
   - `omega_after_clip`

## 兼容性说明
- `reset()` / `update()` 接口未变，保持与 `replay_hierarchical_system.py` 的调用兼容。
- 状态定义未变：`BOOT / STRAIGHTKEEP / APPROACH / TURN / RECOVER`。
- 仅做增量增强，原有状态转移主逻辑保留。

## 快速验证
已完成：
- 语法检查：`python -m py_compile controllers/hierarchical_state_machine.py`
- 行为检查：
  - TURN 超时触发 `turn_timeout` 转移
  - 裁剪字段 `omega_before_clip/omega_after_clip/clip_applied` 正常输出

## 参考示例
```python
from controllers.hierarchical_state_machine import HierarchicalNavigatorStateMachine

sm = HierarchicalNavigatorStateMachine(
    max_turn_steps=20,
    use_fixed_turn_rate=True,
    omega_clip=1.2,
    use_clip=True,
)
```

