# 批量 Replay 结果汇总脚本

> 时间: 2026-03-31 11:09

## 概述

新增 `scripts/collect_replay_batch_results.py`，用于汇总批量 replay 的 `replay_summary.json`，生成系统级总表与简报。

## 新增文件

### `scripts/collect_replay_batch_results.py`

**功能**: 遍历 replay 根目录下所有子目录，读取 `replay_summary.json`，汇总为 CSV + Markdown 总表并打印关键统计。

**命令行参数**:
- `--exp_root`: replay 根目录（必填）
- `--out_csv`: 输出 CSV 路径（默认 `<exp_root>/replay_batch_summary.csv`）
- `--out_md`: 输出 Markdown 路径（默认 `<exp_root>/replay_batch_summary.md`）

**逐 Run 收集字段**（共 17 个）:

| 字段 | 说明 |
|------|------|
| `run_name` | 从 summary.run_dir 提取 |
| `gt_turn_dir` | 真实转向方向 |
| `first_locked_turn_dir` | 首次锁存方向 |
| `most_frequent_locked_turn_dir` | 众数锁存方向 |
| `turn_dir_match` | 方向是否匹配 |
| `num_turn_entries` | 进入 TURN 状态次数 |
| `num_recover_entries` | 进入 RECOVER 状态次数 |
| `first_turn_step` | 首次进入 TURN 的步号 |
| `first_recover_step` | 首次进入 RECOVER 的步号 |
| `turn_duration_steps` | TURN 状态持续总步数 |
| `recover_duration_steps` | RECOVER 状态持续总步数 |
| `num_clip_applied` | omega clip 应用次数 |
| `unique_state_sequence` | 压缩状态链路 |
| `used_valid_only` | 是否仅使用 valid 帧 |
| `original_total_steps` | 原始总帧数 |
| `used_total_steps` | 实际使用帧数 |
| `skipped_invalid_steps` | 跳过的无效帧数 |

**聚合统计指标**（共 8 个）:

| 指标 | 定义 |
|------|------|
| `total_runs` | 总 run 数 |
| `turn_dir_match_rate` | 方向匹配率 |
| `single_turn_success_rate` | num_turn_entries == 1 的比例 |
| `single_recover_success_rate` | num_recover_entries == 1 的比例 |
| `full_sequence_success_rate` | 状态链路含 APPROACH+TURN+RECOVER 的比例 |
| `timeout_trigger_rate` | fallback/timeout 触发率 |
| `mean_turn_duration_steps` | 平均 TURN 持续步数 |
| `mean_recover_duration_steps` | 平均 RECOVER 持续步数 |

**输出**:
1. `replay_batch_summary.csv` — 逐 run 明细表
2. `replay_batch_summary.md` — Markdown 简报（含聚合统计表 + 逐 run 表格）
3. 终端打印关键统计摘要

**示例用法**:
```bash
python scripts/collect_replay_batch_results.py --exp_root results/replay_batch_valid
```

## 设计要点

- **容错性**: 字段缺失时写空值，不报错
- **纯 Python**: 仅依赖标准库（json, csv, os, argparse, datetime），不依赖 ROS2
- **单文件可运行**: 无内部模块依赖
- **中文注释**: 全程使用中文注释
- **聚合统计**: 同时支持终端快速查看和文件归档
