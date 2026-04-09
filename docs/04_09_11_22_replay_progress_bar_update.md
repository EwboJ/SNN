# replay_hierarchical_system.py 进展更新（04_09_11_22）

## 本次目标
在不改变回放逻辑、输出文件格式和统计语义的前提下，为 `run_replay(args)` 的逐帧主循环增加实时进度条，并兼容未安装 `tqdm` 的场景。

## 已完成修改
1. 新增 import
- 新增 `import time`
- 新增 `from collections import deque`
- 新增 `try/except` 导入 `tqdm`，导入失败时 `tqdm = None`

2. 新增辅助函数
- 新增 `_truncate_tail(text, max_len=28)`，用于进度条中裁剪过长 `image_name`，优先保留尾部。

3. 新增计时变量
- 在主循环前新增：
- `replay_start_time = time.perf_counter()`
- `recent_step_times = deque(maxlen=30)`
- `total_frames = len(frames)`

4. 替换主循环
- 将 `for idx, fr in enumerate(frames):` 替换为“可选 tqdm 迭代器”写法：
- 有 `tqdm`：`tqdm(enumerate(frames), total=..., desc='[Replay]', dynamic_ncols=True, unit='frame', smoothing=0.1, leave=True)`
- 无 `tqdm`：自动退化为 `enumerate(frames)`，并在 `verbose=True` 下打印一次警告

5. 新增每帧耗时统计与 postfix
- 每帧开始记录 `step_t0`
- 每帧结束计算 `step_dt`，并写入滚动窗口 `recent_step_times`
- `tqdm` 可用时，新增 `progress.set_postfix(...)` 字段：
- `step`
- `state`（空时 `NA`）
- `lock`（空时 `-`）
- `ms`（滚动平均 ms/frame）
- `fps`（滚动平均 frame/s）
- `img`（裁剪后的图像名）

6. 主循环结束收尾
- `tqdm` 可用时显式 `progress.close()`
- 在 `verbose=True` 下额外打印回放循环耗时与平均速度，不替代原有 `[Replay] 完成` 输出。

## 保持不变项
- 未改动 `argparse` 参数。
- 未改动 `replay_trace.csv`、`replay_summary.json`、`replay_debug.json`、`state_timeline.png` 的生成逻辑与字段结构。
- 未改动既有统计变量含义与 summary 语义。
- 进度条仅包裹逐帧主循环，不包含后处理写文件阶段。

## 验证
- 已执行：`python -m py_compile scripts/replay_hierarchical_system.py`
- 结果：通过（无语法错误）。
