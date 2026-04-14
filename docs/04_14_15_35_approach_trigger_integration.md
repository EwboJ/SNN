# approach_trigger 二分类模型接入层级导航系统

## 时间
2026-04-14 15:35

## 目标
将新训练的 `approach_trigger_sameenv_shortwin` 二分类模型（Straight / NearTurnEvent）接入现有层级导航系统，
让 `STRAIGHTKEEP -> APPROACH` 由 `approach_trigger` 主导触发，而 `stage3` 继续在局部事件区负责 Turn/Recover 证据。

## 修改文件清单

### 1. `inference/corridor_module_infer.py`
- **新增类 `ApproachTriggerInfer`**
  - 继承 `_BaseCorridorInfer`，二分类模型（Straight=0, NearTurnEvent=1）
  - 输出字段与 JunctionLRInfer 完全一致：`pred_label`, `pred_id`, `probs`, `confidence`
  - 默认配置：APLIF_ADD_T4, 48x64
- **`_build_infer()` 注册** `approach_trigger` 模块
- **`main()` choices** 新增 `approach_trigger`

### 2. `scripts/replay_hierarchical_system.py`
- **导入** `ApproachTriggerInfer`
- **模型加载**：从 yaml `models.approach_trigger_ckpt` 读取路径，可选加载
  - 路径存在且文件有效 → 加载模型
  - 路径不存在或未配置 → 输出提示，使用旧逻辑
- **逐帧推理**：trigger 结果传入状态机 `sm.update()` 的 `approach_trigger` 字段
- **trace_csv 新增列**：`trigger_pred`, `trigger_confidence`
- **debug_json 新增**：`approach_trigger` 对象
- **兼容性**：benchmark_only 模式不受影响，旧 yaml 无此字段时完全向后兼容

### 3. `controllers/hierarchical_state_machine.py`
- **`reset()`**：新增 `_trigger_arm_count = 0`
- **`update()` 输入解析**：读取 `approach_trigger` 字段（缺失时为空 dict，兼容旧系统）
- **STRAIGHTKEEP 转移逻辑**（优先路径）：
  - 若 trigger 可用：
    - `NearTurnEvent` 连续 ≥ 2 步
    - 且 (stage3 turn_votes ≥ 2 **或** junction hint 连续一致)
    - → 进入 APPROACH（`approach_trigger_fired`）
  - 若 trigger 不可用或未触发 → 完全保留旧的事件门控逻辑
- **debug 输出新增**：`trigger_pred`, `trigger_confidence`, `trigger_arm_count`
- **不影响 TURN / RECOVER / PROVISIONAL_TURN 等其他状态**

### 4. `configs/hierarchical_nav_event_v2.yaml`（新建）
- 基于 `hierarchical_nav_event_v1b.yaml` 复制所有参数
- 新增 `models.approach_trigger_ckpt` 字段
- 文件头部注释说明用途
- 其余全部参数与 v1b 一致

## 兼容性保证
| 场景 | 行为 |
|---|---|
| 旧 yaml 无 `approach_trigger_ckpt` | replay 正常运行，使用旧事件门控逻辑 |
| 新 yaml 有 `approach_trigger_ckpt` 但路径不存在 | 打印警告，回退旧逻辑 |
| 新 yaml 有 `approach_trigger_ckpt` 且路径有效 | trigger 驱动优先，旧逻辑作为 fallback |
| `benchmark_only` 模式 | 不受影响 |
| `no_save_outputs` 模式 | 不受影响 |

## 验证
- 4 个文件全部通过 Python 语法检查 (`py_compile`)
- yaml 文件可正确解析
- `approach_trigger_ckpt` 路径已正确写入

## 状态
✅ 全部完成
