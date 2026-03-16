# 进展：Pipeline straight_keep 完善

## 完成时间

2026-03-16 15:28

## 修改内容

### `corridor_dataset_pipeline.py`

**新增参数：**
- `--manifest_path`: run manifest CSV 路径（含 group 分组信息）
- `--min_frames`: 最少帧数阈值，低于该值的 run 被跳过（默认 10）
- `--odom_topic`: 里程计话题名称（默认 /odom_raw，上一步已加）

**stage_split 改进：**
- 透传 `manifest_path`、`min_frames`、`exclude` 给 `run_split()`
- straight_keep 时自动将 `group_by` 从 `junction_id,turn_dir` 调整为 `segment_id,direction`
- 自动调整时打印提示

**straight_keep banner 增强：**
- 提示不使用 downsample
- 推荐使用 manifest 按 `segment_id,direction,station_id,condition` 分组划分
- 显示 manifest 是否已指定
- 显示当前 odom 话题配置

## 用法示例

```bash
# straight_keep 全流程（含 manifest）
python scripts/corridor_dataset_pipeline.py \
  --task_type straight_keep \
  --mode all \
  --bag_dir ./data/bags \
  --manifest_path ./data/straight_keep_manifest.csv \
  --min_frames 20 \
  --odom_topic /odom_raw \
  --force

# straight_keep 不含 manifest（按目录名推断分组）
python scripts/corridor_dataset_pipeline.py \
  --task_type straight_keep \
  --mode all \
  --bag_dir ./data/bags \
  --force
```
