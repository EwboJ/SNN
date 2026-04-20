# 推理模块导入诊断增强进展

- 时间戳: 04_20_10_29
- 目标文件: snn_nav_ros/hierarchical_nav_runtime_node.py

## 本次改动

1. 新增 _log_import_failure(import_target, exc) 统一导入失败诊断输出。
2. 在导入异常时输出以下信息：
   - Python executable: ...
   - sys.path[0:5]: ...
   - Import <module> failed: <repr(exc)>
   - 完整 traceback（	raceback.format_exc()）
3. _import_infer_classes() 中异常处理改为记录诊断后 aise 原始异常，不再用通用 RuntimeError 覆盖根因。
4. _import_state_machine_class() 同步采用同样诊断机制，便于区分具体失败 import。

## 日志区分效果

- 仓库根路径解析成功时继续输出：
  - Repository root resolved for imports: <path>
- 具体导入失败时明确输出：
  - Import inference.corridor_module_infer failed: ...
  - 或 Import controllers.hierarchical_state_machine failed: ...

## 兼容性说明

- 未改动节点主控制逻辑与状态机流程。
- 仅增强导入阶段可观测性与错误定位信息。

## 验证

- 运行: python -m py_compile snn_nav_ros/hierarchical_nav_runtime_node.py
- 结果: 通过（语法正确）。
