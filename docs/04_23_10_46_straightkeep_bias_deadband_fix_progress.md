# STRAIGHTKEEP 持续同向转向问题修复进展

时间：2026-04-23 10:46（Asia/Shanghai）

## 背景与目标
- 问题表现：`STRAIGHTKEEP` 阶段在车头接近摆正后仍持续输出同向角速度，导致小车一直转向。
- 已知约束：角速度正负号约定与实机一致，不做全局符号翻转。
- 本次目标：仅在 runtime 的 `STRAIGHTKEEP` 控制链路引入 bias 校正、scale 缩放、clip 限幅、deadband 归零，确保“有偏差时纠偏，摆正后归零直行”。

## 代码改动
- 文件：`snn_nav_ros/hierarchical_nav_runtime_node.py`

1. 新增 `robot_control` 参数读取（`__init__`）
- `straight_keep_bias`（默认 `0.0`）
- `straight_keep_scale`（默认 `1.0`，并做 `>=0` 保护）
- `straight_keep_deadband`（默认 `0.0`，并做 `>=0` 保护）

2. 修改 `_compose_control_cmd()` 中 `state == "STRAIGHTKEEP"` 分支
- 按顺序执行：
  - `raw_omega = float(omega_cmd_final)`
  - `bias_corrected_omega = raw_omega - self.straight_keep_bias`
  - `scaled_omega = self.straight_keep_scale * bias_corrected_omega`
  - `angular_z = clip(scaled_omega, -self.angular_clip, self.angular_clip)`
  - `if abs(angular_z) < self.straight_keep_deadband: angular_z = 0.0`
- `APPROACH / RECOVER / TURN / PROVISIONAL_TURN` 逻辑保持原样。
- 状态机未修改。

3. 增强 `/nav/debug`（compact/full）
- 新增字段：
  - `straight_keep_bias`
  - `straight_keep_scale`
  - `straight_keep_deadband`

## 验证
- 已执行：`python -m py_compile snn_nav_ros/hierarchical_nav_runtime_node.py`
- 结果：语法检查通过。

## 预期效果
- 直段存在稳定偏置时，角速度先经 bias 校正后再输出，减小长期单侧转向风险。
- 偏差较小时由 deadband 直接归零角速度，保留线速度直行，减少“摆正后仍慢慢拐”的现象。
