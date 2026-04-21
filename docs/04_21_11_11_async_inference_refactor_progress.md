# 在线节点异步推理重构进展

- 时间: 04_21_11_11
- 目标文件: snn_nav_ros/hierarchical_nav_runtime_node.py

## 改造目标

解决“同步推理阻塞图像回调”问题，将节点从“同步推理 + 同步发布”重构为“图像回调缓存 + 后台异步推理 + 控制定时器仅读缓存输出”。

## 已完成改造

1. 回调并发结构
- 新增 ReentrantCallbackGroup，分别用于图像订阅与控制 timer。
- main() 改为 MultiThreadedExecutor(num_threads=4)，允许回调并发调度。

2. 图像回调
- 图像回调仅执行：
  - latest 图像缓存
  - latest_image_receive_time 更新时间
  - latest_image_stamp 更新时间
- 未在图像回调中执行任何模型推理。

3. 模型缓存结构
- ModuleCache 扩展为：
  - last_output
  - last_update_time
  - usy
  - uture
  - last_run_step

4. 后台异步推理
- 新增 ThreadPoolExecutor 作为推理线程池。
- 控制线程只提交任务，不阻塞等待。
- 若模块 usy=True，本轮不重复提交。
- 推理完成后通过 non-blocking 收割逻辑回填：
  - last_output
  - last_update_time

5. 控制 timer 逻辑
- 控制 timer 执行顺序调整为：
  1) 非阻塞收割已完成 future
  2) 图像超时检查
  3) 调度决策并提交后台任务
  4) 模型缓存超时检查
  5) 读取缓存输出并状态机 update
  6) 发布 cmd/state/debug
- timer 中不再同步执行模型推理。

6. QoS 与 latest-only
- 图像订阅继续使用 qos_profile_sensor_data。
- 继续使用 depth=1，保持 latest-only。

7. /nav/debug 增强
- 已包含并输出：
  - image_received_ok
  - image_age_ms
  - stage3_busy
  - junction_busy
  - straight_keep_busy
  - 	rigger_busy
  - stage3_age_ms
  - junction_age_ms
  - straight_keep_age_ms
  - 	rigger_age_ms
  - an_stage3
  - an_junction
  - an_straight_keep
  - an_trigger

8. 安全逻辑
- 图像超时继续发布零速。
- 模型缓存超时继续发布零速。
- 节点退出继续发布零速，并新增推理线程池关闭逻辑。

## 兼容性确认

以下逻辑保持不变：
- topic 参数覆盖逻辑
- robot_control 速度映射
- state_conditioned_v2 调度规则
- debug 主发布链路

## 验证

- 执行: python -m py_compile snn_nav_ros/hierarchical_nav_runtime_node.py
- 结果: 通过（语法正确）
