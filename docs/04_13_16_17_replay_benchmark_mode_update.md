# 进展更新（04_13_16_17）

## 目标
为 `scripts/replay_hierarchical_system.py` 增加“纯性能基准测试模式”，用于单独测量三模型推理与状态机单步延迟，并支持禁用写盘输出。

## 已完成修改
1. 新增命令行参数：
- `--benchmark_only`
- `--benchmark_warmup`（默认 50）
- `--benchmark_steps`（默认 500）
- `--no_save_outputs`

2. 新增计时统计函数：
- `_percentile(values, q)`
- `_summarize_timing_ms(values)`

3. 主循环新增分模块计时（`time.perf_counter()`）：
- 读图+预处理 `preprocess_ms`
- `stage3_ms`
- `junction_ms`
- `straight_keep_ms`
- `state_machine_ms`
- `total_step_ms`

4. `benchmark_only` 模式行为：
- 正常读取配置/数据与加载三模型、构建状态机
- 仅处理前 `benchmark_steps` 帧
- 前 `benchmark_warmup` 帧仅 warmup 不计统计
- 不写 trace/debug/summary/timeline 文件
- 控制台输出整体与分模块的 avg/p50/p90/p95/max 以及 achieved_hz

5. `--no_save_outputs` 行为：
- 在非 benchmark 模式下同样生效
- 禁止写出 `replay_trace.csv / replay_summary.json / replay_debug.json / state_timeline.png`
- 控制台保留完成提示

## 兼容性
- 保持原有 replay 推理与状态机流程不变。
- 旧命令可继续运行；不传新参数时行为与原有逻辑兼容。

## 校验
- 已执行：`python -m py_compile scripts/replay_hierarchical_system.py`
- 结果：语法通过。
