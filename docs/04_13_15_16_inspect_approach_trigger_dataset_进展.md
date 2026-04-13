# inspect_approach_trigger_dataset 进展

## 时间
- 2026-04-13 15:16

## 新增文件
- `scripts/inspect_approach_trigger_dataset.py`

## 实现内容
1. 统计功能
- 全局样本数
- train/val/test 各 split 样本数
- Straight / NearTurnEvent 全局分布
- 每个 split 的类别分布
- 每个 run 样本数 TopN（终端打印）

2. 预览图功能
- 随机抽样 run（`--max_runs`）
- 每个 run 均匀抽帧（`--frames_per_run`）
- 上半部分：时间顺序图像 strip
- 下半部分：label_name 彩条 + t_rel_ms 文本
- 边框颜色：NearTurnEvent 橙红系、Straight 蓝绿系
- 输出到 `out_dir/previews/`

3. 参数支持
- `--data_root`
- `--out_dir`
- `--split`（支持单个或逗号分隔）
- `--max_runs`（默认 8）
- `--frames_per_run`（默认 20）
- `--seed`（默认 42）

4. 健壮性
- 缺 `labels.csv` / `images/`：warning 并跳过 run
- 缺 `meta.json`：warning 但不中断
- 缺图像：使用占位图，不中断
- 时间戳 / 数值解析失败：安全处理
- 额外输出 `inspect_summary.json`

5. 绘图库兼容
- matplotlib 改为延迟导入
- 统计功能可独立运行
- 绘图环境不可用时给出 warning，避免脚本直接崩溃

## 验证
- `python -m py_compile scripts/inspect_approach_trigger_dataset.py` 通过
- `python scripts/inspect_approach_trigger_dataset.py --help` 可正常显示
