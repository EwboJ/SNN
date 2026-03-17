# 进展：Pipeline 集成 manifest 自动生成

## 完成时间

2026-03-17 10:25

## 修改文件

### `scripts/corridor_dataset_pipeline.py`

**新增参数：**
| 参数 | 说明 |
|------|------|
| `--auto_build_manifest` | split 前自动从 run 名生成 manifest |
| `--manifest_root` | manifest 输出目录 (默认 split_root 上一级) |
| `--manifest_name` | manifest 文件名 (默认 straight_keep_manifest.csv) |
| `--allow_unknown_manifest` | 无法解析的目录跳过不报错 |
| `--build_manifest_only` | 只生成 manifest 后退出 |
| `--val_ratio` | 验证集比例 (split_mode=ratio) |
| `--test_ratio` | 测试集比例 (split_mode=ratio) |

**修正：**
- straight_keep 默认 `group_by` 从 `segment_id,direction` 改为 `segment_id,condition`

**新增函数：**
- `stage_build_manifest(args)` — 通过 import 直接调用 `build_straight_keep_manifest.build_manifest()`

**主流程改动：**
- `build_manifest_only` 模式：只生成 manifest 后退出
- 自动在 split 前插入 `build_manifest` 阶段（条件：task_type=straight_keep + auto_build_manifest + 无显式 manifest_path）
- banner 增加 manifest 来源 / group_by / 自动生成提示
- `pipeline_log.json` 增加 `manifest` 字段（auto_build_manifest, manifest_path, manifest_generated, manifest_source）
- `stage_split` 传递 `val_ratio` / `test_ratio` 到 `run_split()`

## 用法示例

```bash
# 从 bag 一键全流程
python scripts/corridor_dataset_pipeline.py \
  --task_type straight_keep --mode all \
  --bag_dir ./data/straight_keep_bags \
  --auto_build_manifest \
  --group_by segment_id,condition \
  --split_mode ratio --val_ratio 0.2 --test_ratio 0.2 \
  --copy_mode symlink --force

# 只生成 manifest
python scripts/corridor_dataset_pipeline.py \
  --task_type straight_keep --build_manifest_only \
  --export_root ./data/straight_keep_all
```
