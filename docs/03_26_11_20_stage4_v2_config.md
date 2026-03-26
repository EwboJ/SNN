# 进展：derive_stage1_datasets.py — Stage4 v2 配置增强

## 完成时间

2026-03-26 11:20

## 修改文件

### `scripts/derive_stage1_datasets.py`

Stage4 派生逻辑升级为 v2，支持更灵活的参数控制。

**新增辅助函数：**
- `_sample_phase(frames, max_frames, policy)` — 按策略裁剪帧列表
  - `tail`: 保留尾部（最靠近下一阶段）
  - `uniform`: 均匀抽样
  - `uniform_tail`: 前半均匀 + 后半全保留

**重写 `derive_stage4()` (v2)：**
- 每阶段独立 max_frames 裁剪
- Follow 默认 tail，Turn/Recover 默认 uniform
- margin 参数化（不再写死 300ms）
- 新增 drop_runs_without_follow 检查
- 新增 min_follow_frames 检查
- 返回 info 含 raw_counts + derived_counts

**新增 argparse 参数 (Stage4 v2 参数组)：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage4_pre_turn_ms` | None | 专用 pre_turn_ms |
| `--stage4_recover_ms` | None | 专用 recover_ms |
| `--stage4_turn_margin_ms` | 300 | Turn 边界 margin |
| `--stage4_max_follow_frames` | None | Follow 最多帧 |
| `--stage4_max_approach_frames` | 0 | Approach 最多帧 |
| `--stage4_max_turn_frames` | 0 | Turn 最多帧 |
| `--stage4_max_recover_frames` | 0 | Recover 最多帧 |
| `--stage4_sample_policy` | tail | 裁剪策略 |
| `--stage4_drop_runs_without_follow` | false | 跳过无 Follow 的 run |
| `--stage4_min_follow_frames` | 0 | Follow 最少帧数 |

**dataset_summary.json 增加：**
- `stage4_v2` 节：完整 v2 参数 + `phase_trim_summary`（每阶段 raw/derived 帧数汇总）
- `per_run[].raw_counts` / `per_run[].derived_counts`

**兼容性：**
- 若无 stage4_ 专用参数，自动 fallback 到通用参数
- action3_balanced / junction_lr 逻辑不受影响
