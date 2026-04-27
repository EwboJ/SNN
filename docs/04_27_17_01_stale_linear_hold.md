# 04_27_17_01 慢模型 stale 直行保持进展

## 背景

Jetson 实机调试中，`straight_keep_latency_ms` 约为 830-900 ms，导致 `straight_keep_age_ms` 长时间偏大，`stale_ratio` 约 80%。此前 stale 策略会在角速度过期时同时将 `linear_x` 置零，可能造成地面运行出现间歇式前进。

## 本次更新

- 在 `hierarchical_nav_runtime_node.py` 中新增慢模型条件下的 STRAIGHTKEEP stale linear hold 诊断与控制逻辑。
- `angular.z` 在 STRAIGHTKEEP stale 超时后仍强制置零，避免持续复用旧同向角速度。
- 仅当图像新鲜、`trigger_pred=Straight`、`stage3_pred` 不是 `Turn`、`reason=ok` 且 `straight_keep_age` 未超过短时保持窗口时，允许以低速保持 `linear_x`。
- 新增 `/nav/debug` 字段：`stale_linear_hold_active`、`stale_linear_hold_allowed`、`stale_linear_hold_reason`、`linear_hold_speed_on_stale`、`max_linear_hold_sec`、`linear_hold_max_image_age_sec`。

## 安全边界

- `TURN`、`PROVISIONAL_TURN`、`RECOVER` 不启用 stale linear hold。
- timeout、missing image、exception 分支仍发布全零速度。
- 该机制只影响最终发布的 `Twist`，不改变状态机逻辑、模型输出缓存或 stale 角速度清零策略。
