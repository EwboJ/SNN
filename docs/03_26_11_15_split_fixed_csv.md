# 进展：split_corridor_runs 新增 fixed_split_csv 支持

## 完成时间

2026-03-26 11:15

## 修改文件

### `scripts/split_corridor_runs.py`

新增固定划分文件 (`--fixed_split_csv`) 支持，当提供该参数时直接按 CSV 分配 run 到 train/val/test，跳过 exact/ratio 随机划分。

**新增函数：**
- `load_fixed_split_csv(csv_path)` — 读取固定划分 CSV（至少含 run_name, split），校验 split 值为 train/val/test，禁止重复
- `apply_fixed_split(all_runs, fixed_mapping)` — 按映射分配 + 3 项一致性检查

**一致性检查：**
1. run 不允许重复出现在多个 split
2. 所有有效 run 必须被分配
3. fixed_split_csv 中的 run_name 必须都存在于 src_root

**修改点：**
- `run_split()` 新增 `fixed_split_csv` 参数
- banner 区分 fixed/exact/ratio 模式
- config_out 中 split_mode 为 'fixed'，附带 fixed_split_csv 绝对路径
- split_manifest.csv / split_summary.json 正常输出

## 用法

```bash
python scripts/split_corridor_runs.py \
    --src_root ./data/corridor_balanced \
    --dst_root ./data/corridor \
    --fixed_split_csv ./data/fixed_split.csv
```

**fixed_split.csv 格式：**
```csv
run_name,split
J1_left_r01,train
J1_left_r02,val
J1_left_r03,test
```
