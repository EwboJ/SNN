# replay_hierarchical_system 升级记录（03_30_22_38）

## 更新目标
将 `scripts/replay_hierarchical_system.py` 从基础可视化工具增强为系统级研究分析工具，同时保持原有命令行接口与基本输出链路不变。

## 核心增强项
1. **配置参数对齐（hierarchical_nav.yaml）**
   - 已真正读取并使用：
     - `turn_control.max_turn_steps`
     - `turn_control.use_fixed_turn_rate`
     - `straight_keep.omega_clip`
     - `straight_keep.use_clip`
     - `logging.save_debug_json`
     - `logging.save_csv`
     - `logging.verbose`
   - `_build_state_machine()` 已将上述控制参数传入状态机。

2. **轨迹 CSV 字段增强**
   - `replay_trace.csv` 新增：
     - `gt_phase`
     - `gt_turn_dir`
     - `transition_from`
     - `transition_to`
     - `transition_reason`
     - `clip_applied`
     - `omega_before_clip`
     - `omega_after_clip`
   - `gt_phase` 从 `labels.csv` 的 `phase` 字段读取，无则留空。
   - `gt_turn_dir` 从 `run_name` 推断：`*_left_* -> Left`，`*_right_* -> Right`。

3. **系统级 summary 增强**
   - `replay_summary.json` 新增：
     - `final_locked_turn_dir`
     - `gt_turn_dir`
     - `turn_dir_match`
     - `first_turn_step`
     - `first_recover_step`
     - `turn_duration_steps`
     - `recover_duration_steps`
     - `num_clip_applied`
     - `unique_state_sequence`

4. **逐步 debug 导出**
   - 当 `logging.save_debug_json=true` 时，输出 `replay_debug.json`。
   - 每步至少包含：
     - `step_idx`
     - `state`
     - `transition`
     - `stage_votes`
     - `junction_votes`
     - `recover_support_count`
     - `omega_before_clip`
     - `omega_after_clip`

5. **时间轴图增强**
   - 继续保留四行图结构。
   - 若 `gt_phase` 可得，在 stage3 子图叠加 `gt_phase` marker，便于对比预测与标注。

## 兼容性说明
- 命令行接口未改动：`--run_dir --config --out_dir --device --max_steps`。
- 仍保持基础输出：`replay_trace.csv`、`replay_summary.json`、`state_timeline.png`。
- 不依赖 ROS2。

## 验证结果
- 语法检查通过：`python -m py_compile scripts/replay_hierarchical_system.py`
- 帮助信息可正常输出：`python scripts/replay_hierarchical_system.py --help`

