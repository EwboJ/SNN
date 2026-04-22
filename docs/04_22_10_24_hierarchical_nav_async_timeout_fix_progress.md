# 分层导航异步超时修复进展

- 时间: 2026-04-22 10:24
- 目标文件: `snn_nav_ros/hierarchical_nav_runtime_node.py`
- 需求: 修复首次异步推理在同一 control tick 被立即判定 timeout 的问题。

## 本次修改

1. 修改 `_check_model_timeout_modules()`，在读取 required module 判定信息时增加完整快照字段：
   - `last_update_time`
   - `busy`
   - `future`
   - `last_run_step`
2. 新增 in-flight 宽限窗口（按 step 计算）：
   - `inflight_window_steps = max(1, int(self.model_output_timeout_sec * self.cmd_publish_hz))`
3. timeout 判定改为以下规则：
   - 模块 `busy=True` 且 `self._tick_count - last_run_step <= inflight_window_steps`：暂不判 timeout。
   - 仅在以下情况判 timeout：
     - 模块不 busy 且 `last_update_time is None`
     - 模块不 busy 且 `age_sec > self.model_output_timeout_sec`
     - 模块 busy 且已超过 in-flight 宽限窗口
4. 保持函数签名不变，未修改状态机、图像订阅、线程池与 topic 发布逻辑。

## 校验结果

- 已执行语法检查：`python -m py_compile snn_nav_ros/hierarchical_nav_runtime_node.py`
- 结果: 通过
