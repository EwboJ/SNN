# 进展：新建 build_straight_keep_manifest.py

## 完成时间

2026-03-17 09:52

## 新增文件

### `scripts/build_straight_keep_manifest.py`

从 straight_keep 数据目录中的 run 名自动生成 `runs_manifest.csv`。

**解析格式：**
```
S{segment}_P{station}_{offset}_{yaw}_r{rep}
例: S1_P1_C_Y0_r01, S2_P2_L10_Y0_r03, S4_P1_C_YL5_r02
```

**偏移量映射：**
| 标签 | offset_cm |
|------|-----------|
| C | 0 |
| L10 | -10 |
| R10 | 10 |
| L15 | -15 |
| R15 | 15 |

**航向映射：**
| 标签 | yaw_deg |
|------|---------|
| Y0 | 0 |
| YL5 | -5 |
| YR5 | 5 |

**功能：**
- 支持 flat 和 train/val/test 两种目录结构
- 输出 CSV: run_name, segment_id, station_id, offset_cm, yaw_deg, condition, rep_id
- Segment / Station / Condition 分布统计
- Segment × Condition 交叉表
- 重复 run_name 检查
- `--allow_unknown` 控制未知目录处理

## 用法

```bash
python scripts/build_straight_keep_manifest.py \
    --src_root ./data/straight_keep_all \
    --out_csv ./data/straight_keep_manifest.csv
```
