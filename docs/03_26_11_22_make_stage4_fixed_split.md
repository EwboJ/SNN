# 进展：新建 make_stage4_fixed_split.py

## 完成时间

2026-03-26 11:22

## 新增文件

### `scripts/make_stage4_fixed_split.py`

根据 `stage4_run_manifest.csv` 自动生成固定 train/val/test 划分。

**核心逻辑：**
1. 按 `(junction_id, turn_dir)` 分组
2. 每组按 `pre_turn_ms_available` 排序
3. val 选中位附近，test 选与 val 不同 `delay_bucket`
4. 优先从 `has_all_4_phases=True` + `has_follow=True` 的 run 中选 val/test
5. 候选池逐步放宽：4phases+follow → 4phases → follow → 全部
6. 分组 < 3 runs 时全部归 train 并打印警告

**输出 CSV 字段（8 列）：**
run_name, split, junction_id, turn_dir, pre_turn_ms_available, delay_bucket, has_follow, has_all_4_phases

**用法：**
```bash
python scripts/make_stage4_fixed_split.py \
    --manifest_csv ./data/stage4_run_manifest.csv \
    --out_csv ./data/stage4_fixed_split.csv
```

生成后可直接用于：
```bash
python scripts/split_corridor_runs.py \
    --fixed_split_csv ./data/stage4_fixed_split.csv
```
