# 基于脉冲神经网络的视觉识别与导航算法研究

> **Research on Visual Recognition and Navigation Algorithms Based on Spiking Neural Networks**

本项目围绕脉冲神经网络（SNN）展开两条研究路线：

1. **视觉识别**：在 CIFAR-10/100 基准上，使用 SNN ResNet-110 进行图像分类消融实验（神经元类型、残差模式、时间步长）。
2. **走廊导航**：基于 ROS2 + TurtleBot3 采集走廊场景数据，训练 SNN 策略网络完成自主导航决策。

---

## 目录

- [方法概述](#方法概述)
- [项目目录结构](#项目目录结构)
- [脚本功能说明](#脚本功能说明)
- [环境配置](#环境配置)
- [完整使用流程](#完整使用流程)
- [实验设置](#实验设置)
- [当前进展](#当前进展)
- [后续计划](#后续计划)

---

## 方法概述

### 网络架构

| 组件 | 说明 |
|------|------|
| **骨干网络** | SorResNet-110（基于 ResNet-110 改造的 SNN 版本） |
| **脉冲神经元** | 支持 LIF / PLIF / ALIF / APLIF 四种神经元，通过 `build_neuron()` 统一构建 |
| **残差连接** | 支持 `ADD`（可学习 XOR-like 脉冲残差）和 `standard`（传统残差加法）两种模式 |
| **代理梯度** | 使用 SpikingJelly 的 ATan 代理梯度函数实现反向传播 |

### 视觉识别（CIFAR-10/100）

- 输入图像经 Rate 编码展开为 $T$ 步脉冲序列。
- 每个时间步输入相同图像，最终对所有时间步的输出取平均进行分类。
- 消融变量：`neuron_type` ∈ {LIF, PLIF, ALIF, APLIF}，`residual_mode` ∈ {ADD, standard}，`T` ∈ {2, 4, 8}。

### 走廊导航

- **数据来源**：TurtleBot3 搭载 RealSense 相机在室内走廊中手动遥操作，通过 ROS2 录制 rosbag。
- **输入编码**：支持 Rate 编码和 FrameDiff（帧差编码，模拟事件相机）。
- **策略头**：
  - `DiscreteHead`：分类输出（Forward / Left / Right / Stop / Backward）。
  - `RegressionHead`：回归输出连续角速度 angular_z。
- **在线推理**：支持 `net.step(frame)` 单帧在线推理接口，可直接对接 ROS2 节点。

---

## 项目目录结构

```
cc/
├── README.md                          # 本文件
│
├── # ─── 核心模块 ─────────────────────
├── ADD_ResNet110.py                   # SNN ResNet-110 骨干网络定义
├── neuron_model.py                    # 脉冲神经元实现 (ALIF, APLIF, build_neuron)
├── models/
│   ├── __init__.py
│   └── snn_corridor.py               # 走廊策略网络 (CorridorPolicyNet)
├── datasets/
│   ├── __init__.py
│   ├── corridor_dataset.py            # 走廊数据集加载器 (单帧/序列)
│   └── corridor_task_dataset.py       # Stage1 衍生任务数据集加载器
│
├── # ─── 训练脚本 ─────────────────────
├── train.py                           # 统一训练入口 (CIFAR + 走廊 + Stage1)
├── train_cifar10.py                   # CIFAR-10 消融实验专用训练脚本
│
├── # ─── 评估脚本 ─────────────────────
├── test_cifar10.py                    # CIFAR-10 模型评估 + 混淆矩阵可视化
├── eval_corridor.py                   # 走廊导航模型离线评估
├── eval_spike_stats.py                # CIFAR-10 模型脉冲统计分析
├── eval_corridor_spike_stats.py       # 走廊模型脉冲统计分析
├── verify_alignment.py                # 验证导出数据集的图像-指令对齐质量
│
├── # ─── 数据处理脚本 ─────────────────
├── corridor_export.py                 # 从 ROS2 rosbag 导出走廊数据集
├── corridor_config.yaml               # corridor_export 的配置文件
├── scripts/
│   ├── batch_export.py                # 批量导出多个 rosbag
│   ├── rename_corridor_runs.py        # 重命名走廊数据目录为标准格式
│   ├── split_corridor_runs.py         # 按路口×方向分组划分 train/val/test
│   ├── downsample_corridor.py         # 智能降采样平衡各类别样本
│   ├── extract_loop_windows.py        # 从完整轨迹提取稀疏事件窗口
│   ├── derive_stage1_datasets.py      # 衍生 Stage1 子任务数据集
│   ├── verify_stage1_windows.py       # 可视化验证衍生数据集质量
│   ├── plot_cifar_results.py          # CIFAR 实验结果可视化
│   └── plot_results.py               # 走廊实验结果可视化
│
├── # ─── 数据与输出 ─────────────────
├── data/                              # 数据集目录
│   ├── CIFAR-10/                      #   CIFAR-10 原始数据
│   └── Navigation/                    #   走廊导航数据
├── checkpoint/                        # 训练权重存档
├── results/                           # 评估结果 (图表、指标)
├── analysis_results/                  # 分析输出
├── nav/                               # ROS2 原始 rosbag
│
├── # ─── 文档 ─────────────────────────
└── docs/                              # 开发文档与进度记录
```

---

## 脚本功能说明

### 核心模块

| 文件 | 功能 |
|------|------|
| `ADD_ResNet110.py` | 定义 SNN 版 ResNet-110，包含 `BasicBlock`、`Bottleneck` 和 `SorResNet`。支持 ADD / standard 两种残差模式和多种神经元类型。 |
| `neuron_model.py` | 实现 `ALIFNode`（自适应 LIF）和 `APLIFNode`（自适应参数化 LIF）两种脉冲神经元，提供 `build_neuron()` 工厂函数。 |
| `models/snn_corridor.py` | 走廊策略网络 `CorridorPolicyNet`，复用 SorResNet 骨干，附加 `DiscreteHead` / `RegressionHead`，支持 Rate 和 FrameDiff 编码。 |
| `datasets/corridor_dataset.py` | 走廊导航数据集 `CorridorDataset`（单帧）和 `CorridorSequenceDataset`（时序序列），内置 5→3 动作映射和类别平衡采样。 |
| `datasets/corridor_task_dataset.py` | Stage1 衍生任务数据集 `CorridorTaskDataset`，支持 action3、junction_lr、stage4 等子任务。 |

### 训练脚本

| 文件 | 功能 |
|------|------|
| `train.py` | **统一训练入口**。支持 CIFAR-10/100、走廊导航和 Stage1 子任务三类数据集。包含脉冲监控、TensorBoard 可视化、混合精度训练、类别平衡等功能。 |
| `train_cifar10.py` | CIFAR-10 消融实验专用脚本，精简版训练流程，适合快速跑通基线实验。 |

### 评估脚本

| 文件 | 功能 |
|------|------|
| `test_cifar10.py` | CIFAR-10 模型评估，生成混淆矩阵和分类报告，支持详细脉冲统计分析。 |
| `eval_corridor.py` | 走廊模型离线评估，支持离散分类和连续回归两种模式，生成混淆矩阵和详细指标。 |
| `eval_spike_stats.py` | 对 CIFAR-10 模型进行逐层脉冲统计分析，输出发放率、稀疏度、FPS 等指标。支持多模型批量对比。 |
| `eval_corridor_spike_stats.py` | 对走廊模型进行脉冲统计，含分类精度和混淆矩阵。 |
| `verify_alignment.py` | 验证 `corridor_export.py` 导出数据集的图像与速度指令对齐质量，支持交互式可视化和时间轴分析。 |

### 数据处理脚本（`scripts/`）

| 文件 | 功能 |
|------|------|
| `batch_export.py` | 批量将多个 ROS2 rosbag 导出为走廊数据集，可选择自动划分 train/test。 |
| `rename_corridor_runs.py` | 将旧格式走廊目录名（如 `left1_bag1`）重命名为标准格式（如 `J1_left_r01`）。 |
| `split_corridor_runs.py` | 按路口×方向分组将走廊数据划分为 train/val/test，支持精确指定和比例划分。 |
| `downsample_corridor.py` | 智能降采样：保留全部转弯/停止帧，对直行帧做上下文感知采样和步长采样，缓解类别不平衡。 |
| `extract_loop_windows.py` | 从完整轨迹中提取稀疏事件窗口（路口、阶段、跟随），用于细粒度任务学习。 |
| `derive_stage1_datasets.py` | 从降采样后的走廊数据衍生 Stage1 子任务（action3_balanced、junction_lr、stage4）。 |
| `verify_stage1_windows.py` | 可视化验证衍生 Stage1 数据集的质量和标签一致性。 |
| `plot_cifar_results.py` | CIFAR 实验结果可视化：混淆矩阵、训练曲线、脉冲统计、分类指标。 |
| `plot_results.py` | 走廊实验结果可视化：混淆矩阵、损失/精度曲线、脉冲发放率。 |

---

## 环境配置

### 基础环境

| 依赖 | 版本要求 |
|------|----------|
| Python | >= 3.8 |
| PyTorch | >= 1.12（推荐 2.0+，需支持 CUDA） |
| torchvision | 与 PyTorch 版本对应 |
| SpikingJelly | `clock_driven` 版本（`pip install spikingjelly==0.0.0.0.14`） |
| CUDA | >= 11.6（推荐 11.8 或 12.x） |

### Python 依赖安装

```bash
# 创建虚拟环境（推荐）
conda create -n snn python=3.10 -y
conda activate snn

# 安装 PyTorch（示例：CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 SpikingJelly
pip install spikingjelly==0.0.0.0.14

# 安装其他依赖
pip install numpy matplotlib seaborn scikit-learn tqdm tensorboard pandas opencv-python pillow pyyaml
```

### ROS2 环境（仅走廊数据采集/导出需要）

| 依赖 | 版本要求 |
|------|----------|
| ROS2 | Humble Hawksbill（Ubuntu 22.04） |
| rosbags | `pip install rosbags`（用于离线读取 rosbag，无需完整 ROS2） |

```bash
# 如果仅做离线数据导出，只需安装 rosbags
pip install rosbags

# 如果需要在线采集，请安装完整 ROS2 Humble
# 参考：https://docs.ros.org/en/humble/Installation.html
```

### 硬件建议

- GPU：NVIDIA GPU（显存 >= 8GB，推荐 RTX 3060 及以上）
- 机器人平台（仅数据采集需要）：TurtleBot3 Burger + Intel RealSense 相机

---

## 完整使用流程

### 路线一：CIFAR-10/100 视觉识别

#### Step 1：训练模型

```bash
# 基线实验：APLIF + ADD 残差 + T=4
python train.py --dataset CIFAR10 -T 4 --neuron_type APLIF --residual_mode ADD \
    -b 64 -epochs 150 -lr 0.1 --amp

# 也可以使用专用脚本
python train_cifar10.py -T 4 --neuron_type APLIF --residual_mode ADD \
    -b 64 -epochs 150 -lr 0.1 --amp -enable_tensorboard
```

#### Step 2：评估模型

```bash
# 基础评估（精度 + 混淆矩阵）
python test_cifar10.py -T 4 --neuron_type APLIF --residual_mode ADD \
    -weights checkpoint/CIFAR-10/final_model.ckpt --detailed_spike_stats

# 脉冲统计分析
python eval_spike_stats.py -T 4 --neuron_type APLIF --residual_mode ADD \
    -weights checkpoint/CIFAR-10/final_model.ckpt --export_csv

# 多模型批量对比
python eval_spike_stats.py --batch_compare \
    -T 4 --neuron_type APLIF --residual_mode ADD \
    -weights checkpoint/CIFAR-10/final_model.ckpt
```

#### Step 3：结果可视化

```bash
python scripts/plot_cifar_results.py \
    --ckpt checkpoint/CIFAR-10/final_model.ckpt \
    --tb_dir runs/ \
    --out_dir results/cifar10_baseline/
```

---

### 路线二：走廊导航

#### Step 1：采集数据

使用 TurtleBot3 + ROS2 在走廊中手动遥操作，录制 rosbag：

```bash
# ROS2 录制（在机器人端执行）
ros2 bag record /camera/color/image_raw /cmd_vel -o corridor_bag_01
```

#### Step 2：导出数据集

```bash
# 单个 bag 导出
python corridor_export.py \
    --bag nav/nav_0.db3 \
    --output data/Navigation/raw/ \
    --config corridor_config.yaml

# 批量导出
python scripts/batch_export.py \
    --bag_dir nav/ \
    --output_dir data/Navigation/raw/ \
    --config corridor_config.yaml \
    --split
```

#### Step 3：验证数据对齐

```bash
python verify_alignment.py \
    --data data/Navigation/raw/ \
    --num 20 \
    --output-prefix analysis_results/alignment
```

#### Step 4：数据整理与划分

```bash
# 重命名为标准格式
python scripts/rename_corridor_runs.py --data_dir data/Navigation/raw/

# 按路口×方向分组划分 train/val/test
python scripts/split_corridor_runs.py \
    --src_root data/Navigation/raw/ \
    --dst_root data/Navigation/ \
    --split_mode ratio \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

#### Step 5：数据平衡与降采样

```bash
# 智能降采样（保留转弯帧，稀疏采样直行帧）
python scripts/downsample_corridor.py \
    --src_root data/Navigation/train/ \
    --dst_root data/Navigation/train_ds/ \
    --context_frames 5 \
    --stride 3
```

#### Step 6：衍生 Stage1 子任务（可选）

```bash
# 衍生三类子任务
python scripts/derive_stage1_datasets.py \
    --src_root data/Navigation/train_ds/ \
    --dst_root data/Navigation/stage1/ \
    --task action3_balanced

# 验证衍生数据集
python scripts/verify_stage1_windows.py \
    --data_root data/Navigation/stage1/ \
    --out_dir analysis_results/stage1_verify/
```

#### Step 7：训练策略网络

```bash
# 离散动作分类训练
python train.py --dataset corridor \
    --corridor_root data/Navigation/ \
    -T 4 --neuron_type APLIF --residual_mode ADD \
    -b 32 -epochs 100 -lr 0.01 --class_balance

# Stage1 子任务训练
python train.py --dataset corridor_task \
    --task_root data/Navigation/stage1/ \
    --task_name action3_balanced \
    -T 4 --neuron_type APLIF --residual_mode ADD \
    -b 32 -epochs 100 -lr 0.01
```

#### Step 8：评估与可视化

```bash
# 走廊模型评估
python eval_corridor.py \
    -weights checkpoint/corridor/final_model.ckpt \
    --data_root data/Navigation/ \
    -T 4 --neuron_type APLIF --residual_mode ADD

# 走廊模型脉冲统计
python eval_corridor_spike_stats.py \
    -weights checkpoint/corridor/final_model.ckpt \
    --corridor_root data/Navigation/ \
    -T 4 --neuron_type APLIF --residual_mode ADD

# 结果可视化
python scripts/plot_results.py \
    --ckpt checkpoint/corridor/final_model.ckpt \
    --data_root data/Navigation/ \
    --out_dir results/corridor_baseline/
```

---

## 实验设置

### CIFAR-10 消融实验

| 实验因素 | 变量选项 | 说明 |
|----------|----------|------|
| 神经元类型 | LIF / PLIF / ALIF / APLIF | 从固定阈值到自适应参数化 |
| 残差模式 | ADD / standard | ADD 为可学习脉冲残差 |
| 时间步长 T | 2 / 4 / 8 | 脉冲展开步数 |

**固定超参数：**

| 参数 | 值 |
|------|-----|
| 网络深度 | ResNet-110 |
| 优化器 | SGD (momentum=0.9, weight_decay=5e-4) |
| 初始学习率 | 0.1 |
| 学习率调度 | CosineAnnealingLR |
| 训练轮数 | 150 epochs |
| 批大小 | 64 / 128 |
| 数据增强 | RandomCrop(32, padding=4) + RandomHorizontalFlip |
| 混合精度 | AMP (FP16) |

### 走廊导航实验

| 参数 | 值 |
|------|-----|
| 输入分辨率 | 32 × 32（resize） |
| 动作空间 | 3 类 (Left / Forward / Right) 或 5 类 |
| 编码方式 | Rate / FrameDiff |
| 时间步长 T | 4 / 8 |
| 机器人平台 | TurtleBot3 Burger |
| 相机 | Intel RealSense (640×480, 60° FOV) |
| 走廊宽度 | ~2.0 m |

---

## 当前进展

### 已完成

- [x] SNN ResNet-110 骨干网络实现（ADD / standard 残差模式）
- [x] 四种脉冲神经元实现（LIF / PLIF / ALIF / APLIF）
- [x] CIFAR-10 训练与评估流程
- [x] CIFAR-10 消融实验框架（神经元 × 残差 × 时间步）
- [x] ROS2 rosbag → 走廊数据集自动导出管线
- [x] 数据对齐验证工具
- [x] 智能降采样与类别平衡机制
- [x] 走廊策略网络（DiscreteHead + RegressionHead）
- [x] 走廊数据集加载器（单帧 / 序列 / Stage1 衍生）
- [x] Stage1 衍生任务提取（action3、junction_lr、stage4）
- [x] 脉冲统计分析与可视化工具
- [x] TensorBoard 训练过程监控
- [x] 结果可视化（混淆矩阵、训练曲线、脉冲发放率）

### 初步实验结果

| 实验 | 配置 | Top-1 Acc |
|------|------|-----------|
| CIFAR-10 基线 | APLIF + ADD + T=4 | 待补充 |
| CIFAR-10 | APLIF + ADD + T=8 | 待补充 |

> 详细消融实验结果待完成全部实验后补充。

---

## 后续计划

- [ ] 完成 CIFAR-10 全部消融实验（4 神经元 × 2 残差 × 3 时间步 = 24 组）
- [ ] 走廊导航模型调参与完整训练
- [ ] Stage1 子任务实验验证
- [ ] CIFAR-100 迁移实验
- [ ] FrameDiff 编码 vs Rate 编码对比实验
- [ ] 在线部署：ROS2 节点集成 `net.step()` 实时推理
- [ ] 模型轻量化与推理延迟分析
- [ ] 论文实验表格整理与可视化

---

## 引用

如果本项目对您的研究有帮助，请引用：

```
待补充
```

---

## License

本项目采用 [MIT License](LICENSE) 开源协议。
