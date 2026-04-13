# derive_approach_trigger_dataset t_rel_ms 写出修复进展

## 时间
- 2026-04-13 15:36

## 问题
- 训练读取派生 labels.csv 时，`t_rel_ms=''` 导致 `float('')` 报错。

## 修复内容
1. 新增统一浮点清洗函数 `to_csv_float(v, default=0.0)`：
- `None -> default`
- `'' -> default`
- 非法值 / nan / inf -> default
- 合法值 -> `float(v)`

2. 新增辅助函数：
- `_is_nonempty_parseable_float(v)`：统计“非空且可解析”
- `_format_csv_float(v)`：写出可解析字符串，且至少保留一位小数（如 `0.0`）

3. 修复 labels.csv 写出：
- `t_rel_ms` 统一走 `to_csv_float(..., default=0.0)` + `_format_csv_float(...)`
- NearTurnEvent 保留真实相对时间（负值）
- Straight 空值自动落盘为 `0.0`

4. 统计增强（终端与 summary 都可见）：
- 每个 split 的 `t_rel_ms_nonempty_parseable`
- 每个 split 的 `t_rel_ms_default_filled_0.0`

5. run 级统计也补充：
- `t_rel_ms_nonempty_parseable`
- `t_rel_ms_default_filled`

## 验证
- `python -m py_compile scripts/derive_approach_trigger_dataset.py` 通过
- `to_csv_float` 与 `_format_csv_float` 快速用例验证通过（空值写成 `0.0`）
