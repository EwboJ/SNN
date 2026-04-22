# runtime image timeout 诊断与轻量 debug 改造进展

- 时间: 2026-04-22 14:45
- 目标文件: `snn_nav_ros/hierarchical_nav_runtime_node.py`
- 目标: 提升图像接收与异步调度诊断能力，并在默认场景下降低 `/nav/debug` 序列化与发布开销。

## 本次修改

1. 图像接收计数
- 在 `__init__` 新增 `self._image_rx_count = 0`。
- 在 `_image_callback()` 中仅当图像转换成功并写入 latest image 缓存后执行 `self._image_rx_count += 1`。

2. 诊断字段增强
- 在 `_publish_debug()` 新增并输出：
  - `tick_count`
  - `image_rx_count`
  - `stage3_last_run_step`
  - `junction_last_run_step`
  - `straight_keep_last_run_step`
  - `trigger_last_run_step`
- 字段来源分别为：
  - `self._tick_count`
  - `self._image_rx_count`
  - `self.module_cache[...].last_run_step`

3. 轻量 debug 模式（默认开启）
- 新增配置读取开关：`self.debug_compact = bool(self.safety_cfg.get("debug_compact", True))`。
- 当 `debug_compact=True`：
  - `_publish_debug()` 仅发布核心字段集合。
  - 不再写入 `state_machine_debug`。
  - 不再附加 `extra` 扩展字段，以控制 payload 体积。
- 当 `debug_compact=False`：
  - 保持完整 debug 行为（含 `state_machine_debug` 与 `extra`）。

## 兼容性与边界

- 未修改图像订阅流程与图像 timeout 判定逻辑。
- 未修改异步推理调度与回收逻辑。
- 未修改安全停车与状态机更新逻辑。

## 校验

- 已执行语法检查：`python -m py_compile snn_nav_ros/hierarchical_nav_runtime_node.py`
- 结果: 通过
