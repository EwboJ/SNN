# plot_results 兼容性改造进展

## 时间
- 2026-04-13 11:41

## 目标
- 提升 `scripts/plot_results.py` 对 corridor_task 自定义 `task_name` 的兼容性。
- 保持既有输出流程不变（confusion_matrix、per_class_metrics、metrics.json、run_predictions.csv 等）。

## 已完成修改
1. 新增标准任务模板映射 `DEFAULT_TASK_CLASS_NAMES`，作为归一化后任务名的模板来源。
2. 新增 `normalize_task_name(task_name)`：
- 小写化
- `-` 统一为 `_`
- 去首尾空格
- 前缀归一：`stage3* -> stage3`、`stage4* -> stage4`、`junction_lr* -> junction_lr`、`action3_balanced* -> action3_balanced`
- 无法归一时返回清洗后的原值。
3. 新增 `parse_cli_class_names(cli_class_names)`：
- 支持逗号分隔字符串解析
- 自动 `strip`
- 空值返回 `None`
4. 新增 `resolve_class_names(task_name, num_actions, cfg=None, data_root=None, cli_class_names=None)`，解析优先级：
- `cfg['class_names']`
- `--class_names`
- 归一化任务名模板映射
- `class_i` 自动回退
5. 新增 `_ensure_class_names_for_count(...)` 与 `_generate_generic_class_names(...)`：
- 当 `class_names` 与类别数不一致时告警并回退到 `class_0..class_n-1`，避免混淆矩阵和 per-class 绘图崩溃。
6. 新增 `--class_names` 参数：
- 支持示例：`--class_names Approach,Turn,Recover`
- 长度不匹配时警告并忽略，继续自动解析。
7. corridor_task 分支改为调用 `resolve_class_names(...)`，不再依赖固定任务名精确匹配。
8. 日志输出增强：
- `task_name_raw`
- `task_name_normalized`
- `class_names`
9. `metrics.json` 元信息改造：
- 保留原始任务名，不再强制覆盖为旧主任务名
- 新增字段：`task_name_raw`、`task_name_normalized`
- 兼容附加字段：`normalized_task_name`
- `task_name` 现在写入原始/保守推断后的 raw 值。
10. 仅在 `task_name` 缺失时做保守推断（基于 `data_root + num_actions`），并对推断结果做归一化。

## 验证
- 执行：`python -m py_compile scripts/plot_results.py`
- 结果：通过

## 影响说明
- 推理、混淆矩阵计算、per-class 统计、run-level 汇总、结果导出路径与文件名保持不变。
- 变更聚焦于 `task_name/class_names` 解析兼容性与 metadata 写出增强。
