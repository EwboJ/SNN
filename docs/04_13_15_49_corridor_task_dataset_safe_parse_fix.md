# corridor_task_dataset 数值解析安全修复进展

## 时间
- 2026-04-13 15:49

## 修改文件
- `datasets/corridor_task_dataset.py`

## 背景问题
- `_load_task_labels_csv()` 中存在直接 `float(...)` / `int(...)` 解析。
- 当 csv 中出现 `t_rel_ms=''` 等脏值时，构建数据集阶段抛出：
  - `ValueError: could not convert string to float: ''`

## 本次修复
1. 在 `_load_task_labels_csv()` 内新增安全转换函数：
- `_safe_float(v, default=0.0)`
- `_safe_int(v, default=0)`

2. `_safe_float` 行为：
- `None -> default`
- `'' / 仅空格 -> default`
- 非法字符串 / nan / inf -> default
- 合法数值 -> `float(v)`

3. `_safe_int` 行为：
- `None / '' / 仅空格 / 非法值 -> default`
- 合法整数或浮点字符串 -> `int(...)`

4. `_load_task_labels_csv()` 中所有来自 csv 的数值字段改为安全解析：
- `label_id`
- `timestamp_ns`
- `linear_x`
- `angular_z`
- `valid`
- `orig_action_id`
- `t_rel_ms`

5. 关键字段损坏行保护：
- 关键字段：`image_name` / `label_id` / `timestamp_ns`
- 任一损坏：warning + 跳过该行（不崩溃）
- 文件结束后打印跳过行总数

## 兼容性
- 保持 `CorridorTaskDataset` 输出结构和训练接口不变。
- 仅增强 csv 脏数据容错能力。

## 验证
- `python -m py_compile datasets/corridor_task_dataset.py` 通过。
