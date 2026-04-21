# 图像订阅与 Topic 参数逻辑修复进展

- 时间: 04_21_10_39
- 修改文件: snn_nav_ros/hierarchical_nav_runtime_node.py

## 本次目标

1. 明确 topic 参数覆盖优先级（launch > yaml > default）。
2. 图像订阅 QoS 对齐 sensor_data，并保持 latest-only 缓存。
3. 增强图像接收诊断和 /nav/debug 字段可观测性。

## 已完成修改

1. 显式参数声明保留并确认：
   - config_path
   - image_topic
   - cmd_vel_topic
   - state_topic
   - debug_topic
2. 新增 _resolve_topic_value()，统一按 launch > yaml > default 解析 4 个 topic。
3. 启动日志新增最终 topic 与来源打印（launch/yaml/default）。
4. 图像订阅 QoS 改为基于 qos_profile_sensor_data，并设置 depth=1（latest-only）。
5. 图像回调补充记录：
   - self._latest_image_stamp
   - self._latest_image_receive_time
6. 在图像缺失/超时时增加节流告警：
   - 当前订阅 topic
   - 最近一次成功接收时间
   - 最近 header stamp
7. /nav/debug 增加字段：
   - subscribed_image_topic
   - image_received_ok
   - image_age_ms（原有保留）
   - image_header_stamp
   - latest_image_receive_time（附加）

## 未改动范围

- state_conditioned_v2 调度逻辑未改。
- 模型缓存逻辑未改。
- 安全停车逻辑未改。

## 验证

- 执行：python -m py_compile snn_nav_ros/hierarchical_nav_runtime_node.py
- 结果：通过（语法正确）。
