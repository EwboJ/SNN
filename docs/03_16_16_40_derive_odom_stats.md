# 进展：derive_straight_keep 增强 odom 统计

## 完成时间

2026-03-16 16:40

## 修改内容

### `scripts/derive_straight_keep_dataset.py`

**新增函数：**

| 函数 | 说明 |
|------|------|
| `read_odom_csv(path)` | 读取 `odom_raw.csv`，返回里程计记录列表或 None |
| `compute_odom_stats(records, t_start, t_end)` | 计算 yaw 变化、路径长度、角速度统计 |

**`compute_odom_stats` 输出字段：**
- `odom_points` — 有效 odom 记录数
- `odom_duration_s` — 时间跨度
- `yaw_net_change_rad` — yaw 净变化（处理环绕）
- `yaw_abs_change_rad` — yaw 累计绝对变化
- `yaw_range_rad` / `yaw_min_rad` / `yaw_max_rad`
- `path_length_m` — 累积路径长度
- `angular_w_mean` / `angular_w_abs_mean` — 角速度统计

**`process_run()` 改动：**
- 读取 `odom_raw.csv`（可选），按输出帧时间窗口裁剪
- info dict 新增 `odom_stats` 字段

**`write_derived_run()` 改动：**
- meta.json 增加 `odom_available` (bool) 和 `odom_stats` (dict)

**`run_derive_straight_keep()` 改动：**
- 每个 run 的终端输出追加 odom 标签 (Δyaw, path)
- dataset_summary.json 新增 `odom_summary` 汇总统计
- 总结区域新增 odom 统计表（yaw 净/累计变化、路径长度、平均角速度）
- 无 odom 时打印 "无 odom" 提示

**兼容性：**
- 无 `odom_raw.csv` → `odom_available=false`，逻辑完全不变
- phase 判定仅依据 angular_z，odom 仅作为附加统计
