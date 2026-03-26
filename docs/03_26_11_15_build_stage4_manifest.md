# 进展：新建 build_stage4_run_manifest.py

## 完成时间

2026-03-26 11:15

## 新增文件

### `scripts/build_stage4_run_manifest.py`

为 stage4 四阶段数据集构建 run 级 manifest，含 turn 事件检测和阶段帧数统计。

**支持的 run 名格式：**
| 格式 | 示例 | 解析 |
|------|------|------|
| 标准 junction | `J1_left_r03` | junction_id=1, turn_dir=left, rep_id=3 |
| 旧版兼容 | `left1_bag3` | junction_id=1, turn_dir=left, rep_id=3 |

**Turn 检测逻辑（复用 derive_stage1）：**
1. action_name 连续 K 帧为 Left/Right（优先）
2. |angular_z| >= threshold 且符号一致连续 K 帧

**输出 CSV 字段：**
run_name, junction_id, turn_dir, rep_id, total_frames, valid_frames,
t_turn_on_ns, t_turn_off_ns, turn_duration_ms, pre_turn_ms_available,
follow/approach/turn/recover_frames_raw, has_follow, has_all_4_phases,
delay_bucket, turn_detect_method

**delay_bucket 分桶：**
very_short (<500ms), short (500-1500ms), medium (1500-3000ms),
long (3000-5000ms), very_long (5000ms+), no_turn

**统计输出：** Junction 分布、Turn Direction 分布、四阶段帧数汇总、
delay 分桶分布、四阶段完整率。

## 用法

```bash
python scripts/build_stage4_run_manifest.py \
    --src_root ./data/corridor_all \
    --out_csv ./data/stage4_run_manifest.csv \
    --out_json ./data/stage4_run_manifest.json \
    --allow_unknown
```
