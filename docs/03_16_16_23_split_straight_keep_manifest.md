# 进展：split_corridor_runs 支持 straight_keep manifest

## 完成时间

2026-03-16 16:23

## 修改内容

### `scripts/split_corridor_runs.py`

**`load_manifest()` — 通用 manifest 读取**
- 返回 `(manifest_dict, field_names)` 元组
- 自动读取 CSV 全部字段，不再硬编码 junction_id/turn_dir/rep_id
- 新增 `_auto_convert()` 自动转换 int/float/str

**`build_groups()` — 任意字段分组**
- 新增 `manifest_fields` 参数
- 支持 `group_by=segment_id,direction,station_id` 等任意组合

**`save_split_manifest()` — 动态列输出**
- 新增 `manifest_fields` 参数
- 有 manifest 时输出全部字段列
- 无 manifest 时回退到 junction_id/turn_dir/rep_id（兼容）

**`run_split()` — 增强日志**
- 打印 manifest 字段列表
- 检查 group_keys 是否在 manifest 字段中
- 缺失时警告并提示可用字段

**兼容性：**
- junction 数据（无 manifest）行为不变
- straight_keep manifest 示例字段：
  run_name, segment_id, direction, station_id, offset_cm, yaw_deg, target_speed_mps, rep_id

## 用法

```bash
# junction 传统用法（不受影响）
python scripts/split_corridor_runs.py \
  --src_root ./data/corridor_balanced \
  --dst_root ./data/corridor

# straight_keep 使用 manifest
python scripts/split_corridor_runs.py \
  --src_root ./data/straight_keep_all \
  --dst_root ./data/straight_keep \
  --manifest ./straight_keep_manifest.csv \
  --group_by segment_id,direction,station_id \
  --split_mode ratio --val_ratio 0.2 --test_ratio 0.2
```
