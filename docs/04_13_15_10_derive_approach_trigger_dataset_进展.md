# approach_trigger 数据集派生脚本进展

## 时间
- 2026-04-13 15:10

## 新增文件
- `scripts/derive_approach_trigger_dataset.py`

## 目标完成情况
- 已实现从 `./data/corridor_stage3_rawsplit` 派生二分类数据集 `approach_trigger_v1_sameenv_shortwin`。
- 标签定义：
  - `Straight` -> `label_id=0`
  - `NearTurnEvent` -> `label_id=1`

## 核心规则实现
1. turn 事件检测
- 连续 `Left/Right` 且同方向帧段长度 `>= min_turn_k` 记为 turn 事件。
- 记录 `turn_dir/turn_on_ns/turn_off_ns/idx_on/idx_off`。

2. 正样本 NearTurnEvent
- 取窗口 `[turn_on - pre_turn_ms, turn_on - pre_turn_end_ms]`（默认 `[700ms,100ms]` 之前）。
- 窗口内有效帧标为正样本，写出 `t_rel_ms`（相对 `turn_on`，负值）。

3. 负样本 Straight
- 仅从 `Follow/Forward` 帧中选。
- 与任意 turn 事件距离 `>= safe_margin_ms`（默认 2500ms）。
- 负样本按 `stride` 抽样（默认 3）。

4. 忽略区
- turn 本身
- `turn_off` 后 `post_turn_exclude_ms` 内（默认 1000ms）
- 不属于正样本且又未达到负样本安全边界的中间帧

## I/O 与写出
- 输出目录结构：`dst_root/{train,val,test}/{run_name}/`
- 每个 run 输出：
  - `images/`
  - `labels.csv`
  - `meta.json`
- 额外输出：`dst_root/derive_summary.json`

## labels.csv 字段
- `image_name`
- `label_id`
- `label_name`
- `orig_action_name`
- `timestamp_ns`
- `t_rel_ms`
- `valid`
- `source_run`
- `source_split`

## meta.json 字段
- `dataset_name`
- `split`
- `run_name`
- `total_frames`
- `label_distribution`
- `source_type`（sameenv 或 mixed）
- `derive_config`（参数快照）
- `detected_turn_count`
- `turn_summary`

## 可选 straight_root
- 已保留 `--straight_root` 参数，默认空（不启用）。
- 启用后读取其 train/val/test 作为辅助 Straight 负样本来源。
- 在 run 的 `meta.json` 记录辅助来源信息。

## 命令行参数
- 已支持：
  - `--src_root`
  - `--dst_root`
  - `--straight_root`
  - `--pre_turn_ms`
  - `--pre_turn_end_ms`
  - `--safe_margin_ms`
  - `--post_turn_exclude_ms`
  - `--stride`
  - `--min_turn_k`
  - `--copy_mode copy|symlink`
  - `--valid_only`
  - `--force`

## 边界与健壮性
- run 无 turn：跳过并 warning。
- 缺 `labels.csv` 或 `images/`：跳过并 warning。
- 时间戳/关键字段解析失败：安全跳过该行。

## 验证
- `python -m py_compile scripts/derive_approach_trigger_dataset.py` 通过。
- `python scripts/derive_approach_trigger_dataset.py --help` 可正常显示参数。
