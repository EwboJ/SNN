"""
走廊导航策略网络 — 基于 SNN ResNet Backbone
================================================
复用 ADD_ResNet110.py 的 SorResNet 作为视觉骨干，
新增离散/回归输出头 + rate/framediff 编码 + 序列/在线推理接口。

不破坏原 CIFAR-10 分类训练：这是全新的策略网络文件。

示例:
    from models.snn_corridor import CorridorPolicyNet, build_corridor_net

    # 离散动作 (3类: Left/Straight/Right), rate 编码
    net = build_corridor_net(
        head_type='discrete', num_actions=3, encoding='rate',
        T=8, neuron_type='APLIF', residual_mode='ADD')

    # 连续控制 (angular_z), framediff 编码
    net = build_corridor_net(
        head_type='regression', control_dim=1, encoding='framediff',
        T=1, neuron_type='APLIF', residual_mode='ADD')

    # ROS 在线推理 (framediff)
    net.reset_state()
    for frame in camera_stream:
        action = net.step(frame)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional
from typing import Optional, Union, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ADD_ResNet110 import SorResNet, BasicBlock


# ============================================================================
# 帧差编码器
# ============================================================================
class FrameDiffEncoder(nn.Module):
    """
    帧差事件化编码器。

    将连续 RGB 图像转为类事件表示：
      x_pos = ReLU(I_t - I_{t-1})    # 亮度增加 → ON 事件
      x_neg = ReLU(I_{t-1} - I_t)    # 亮度减少 → OFF 事件
      output = concat(x_pos, x_neg)  # [B, 2*C_in, H, W] (C_in=3 → 6通道)

    首帧时 prev_frame 为 None → 差分为 0 → 输出全零。
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2  # pos + neg
        # prev_frame 用 register_buffer 以便 .to(device) 能自动移动
        self.register_buffer('prev_frame', None)

    def set_prev_frame(self, frame: torch.Tensor):
        """
        手动设置前一帧 (用于 ROS 在线推理初始化)。

        Args:
            frame: [B, C, H, W] 或 [C, H, W]
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        self.prev_frame = frame.detach().clone()

    def reset(self):
        """清除前一帧状态"""
        self.prev_frame = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 当前帧

        Returns:
            [B, 2*C, H, W] 帧差编码
        """
        if self.prev_frame is None:
            # 首帧：差分为 0
            diff_pos = torch.zeros_like(x)
            diff_neg = torch.zeros_like(x)
        else:
            prev = self.prev_frame
            # 确保 batch 维度匹配（首帧可能和后续帧 batch 不同）
            if prev.shape[0] != x.shape[0]:
                prev = prev[:x.shape[0]]
            diff_pos = F.relu(x - prev)
            diff_neg = F.relu(prev - x)

        # 更新 prev_frame
        self.prev_frame = x.detach().clone()

        return torch.cat([diff_pos, diff_neg], dim=1)


# ============================================================================
# 输出头
# ============================================================================
class DiscreteHead(nn.Module):
    """离散动作分类头：backbone 特征 → logits"""

    def __init__(self, feat_dim: int, num_actions: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_actions, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)  # [B, num_actions]


class RegressionHead(nn.Module):
    """
    连续控制回归头：backbone 特征 → 控制量

    可选 tanh 限幅 + scale：
      - angular_z: tanh * w_max (e.g. 2.84 rad/s for TurtleBot)
      - linear_x:  tanh * v_max (e.g. 0.22 m/s for TurtleBot)
    """

    def __init__(self, feat_dim: int, control_dim: int = 1,
                 use_tanh: bool = True,
                 v_max: float = 0.22, w_max: float = 2.84,
                 dropout: float = 0.3):
        super().__init__()
        self.control_dim = control_dim
        self.use_tanh = use_tanh
        self.v_max = v_max
        self.w_max = w_max

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, control_dim, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.head(features)  # [B, control_dim]
        if self.use_tanh:
            out = torch.tanh(out)
            if self.control_dim == 1:
                out = out * self.w_max  # angular_z only
            else:
                # [v, w]: 分别缩放
                scales = torch.tensor(
                    [self.v_max, self.w_max],
                    device=out.device, dtype=out.dtype
                )
                out = out * scales.unsqueeze(0)
        return out


# ============================================================================
# CorridorPolicyNet
# ============================================================================
class CorridorPolicyNet(nn.Module):
    """
    走廊导航策略网络。

    architecture:
        [input] → [encoding] → [SNN backbone (无原 fc)] → [feature] → [head] → [action]

    encoding 模式:
        rate:       单帧重复 T 次（backbone 内部处理），in_channels 不变
        framediff:  帧差编码，in_channels 变为 2*原始通道

    head 类型:
        discrete:   分类 → logits [B, num_actions]
        regression: 回归 → [B, control_dim] (可选 tanh 限幅)

    推理模式:
        单帧:   forward(x)       x: [B, C, H, W]
        序列:   forward_seq(xs)  xs: [B, L, C, H, W] → [B, L, *]
        在线:   step(frame)      frame: [B, C, H, W] (ROS 实时推理)
    """

    def __init__(
        self,
        backbone: SorResNet,
        head_type: str = 'discrete',
        num_actions: int = 3,
        control_dim: int = 1,
        encoding: str = 'rate',
        raw_in_channels: int = 3,
        use_tanh: bool = True,
        v_max: float = 0.22,
        w_max: float = 2.84,
        dropout: float = 0.3,
    ):
        """
        Args:
            backbone: SorResNet 实例 (已创建，in_channels 已匹配 encoding)
            head_type: 'discrete' 或 'regression'
            num_actions: 离散动作数 (3 或 5)
            control_dim: 回归维度 (1 或 2)
            encoding: 'rate' 或 'framediff'
            raw_in_channels: 原始图像通道数 (通常 3=RGB)
            use_tanh: 回归头是否用 tanh 限幅
            v_max: 最大线速度 (m/s)
            w_max: 最大角速度 (rad/s)
            dropout: head 的 dropout 概率
        """
        super().__init__()
        assert head_type in ('discrete', 'regression')
        assert encoding in ('rate', 'framediff')

        self.head_type = head_type
        self.encoding = encoding
        self.raw_in_channels = raw_in_channels

        # ===== 编码器 =====
        self.frame_diff_encoder = None
        if encoding == 'framediff':
            self.frame_diff_encoder = FrameDiffEncoder(raw_in_channels)

        # ===== Backbone (去掉原始 fc, 保留特征提取) =====
        self.backbone = backbone
        # 获取 backbone 特征维度
        # SorResNet 的 fc 输入维度 = 128 * BasicBlock.expansion = 128
        self.feat_dim = 128 * BasicBlock.expansion

        # ===== 输出头 =====
        if head_type == 'discrete':
            self.head = DiscreteHead(self.feat_dim, num_actions, dropout)
            self.num_actions = num_actions
        else:
            self.head = RegressionHead(
                self.feat_dim, control_dim, use_tanh, v_max, w_max, dropout)
            self.control_dim = control_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入编码。

        Args:
            x: [B, C_raw, H, W] 原始图像

        Returns:
            [B, C_backbone, H, W] 编码后的输入
        """
        if self.encoding == 'framediff':
            return self.frame_diff_encoder(x)
        return x  # rate: 直接传入

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        用 backbone 提取特征 (使用 return_features=True，跳过原始 fc)。

        Args:
            x: [B, C_backbone, H, W] 编码后的输入

        Returns:
            features: [B, feat_dim]
        """
        _, features = self.backbone(x, return_features=True)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        单帧前向传播。

        Args:
            x: [B, C_raw, H, W] 原始图像

        Returns:
            discrete:   logits [B, num_actions]
            regression: y [B, control_dim]
        """
        encoded = self._encode(x)
        features = self._extract_features(encoded)
        return self.head(features)

    def forward_seq(self, xs: torch.Tensor,
                    reset_at_start: bool = True) -> torch.Tensor:
        """
        序列前向传播 (用于训练 / CorridorSequenceDataset)。

        SNN 状态在序列开始时可选 reset，之后**跨帧保留状态**。

        framediff 模式:
          - 序列第0帧的 prev_frame 为 None → 差分为 0
          - 后续帧与前帧做差分，SNN 状态自然累积

        rate 模式:
          - 每帧独立通过 backbone (backbone 内部已做 T 步展开)
          - 帧间 SNN 状态保留 (backbone 内部 reset 只在最外层调用)

        Args:
            xs: [B, L, C_raw, H, W] 帧序列
            reset_at_start: 是否在序列开始时 reset SNN 状态

        Returns:
            outputs: [B, L, num_actions] 或 [B, L, control_dim]
        """
        B, L, C, H, W = xs.shape

        if reset_at_start:
            self.reset_state()

        outputs = []
        for t in range(L):
            frame = xs[:, t]  # [B, C, H, W]
            out = self.forward(frame)
            outputs.append(out)

            # rate 模式：每帧推理后需要 reset backbone SNN 状态
            # (因为 backbone 内部 T 步展开是完整的脉冲过程)
            if self.encoding == 'rate':
                functional.reset_net(self.backbone)

            # framediff 模式：不 reset，让 SNN 状态跨帧累积

        return torch.stack(outputs, dim=1)  # [B, L, *]

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        """
        在线推理：每来一帧推一次 (ROS 实时推理用)。

        用法:
            net.reset_state()
            for frame in camera_stream:
                action = net.step(frame)  # frame: [1, C, H, W]

        注意:
          - rate 模式下，每次 step 后自动 reset backbone SNN 状态
          - framediff 模式下，SNN 状态跨帧累积，不自动 reset

        Args:
            frame: [B, C_raw, H, W] 或 [C_raw, H, W]

        Returns:
            discrete:   logits [B, num_actions]
            regression: y [B, control_dim]
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        out = self.forward(frame)

        if self.encoding == 'rate':
            functional.reset_net(self.backbone)

        return out

    def reset_state(self):
        """
        完全重置所有 SNN 状态。

        包括：
          - backbone 中所有脉冲神经元的膜电位
          - framediff 编码器的 prev_frame
        """
        functional.reset_net(self)
        if self.frame_diff_encoder is not None:
            self.frame_diff_encoder.reset()

    def set_prev_frame(self, frame: torch.Tensor):
        """
        手动设置 framediff 编码器的前一帧。

        用于 ROS 在线推理时，如果希望第一帧就有有效差分，
        可以先设置 prev_frame。

        Args:
            frame: [B, C, H, W] 或 [C, H, W]
        """
        if self.frame_diff_encoder is None:
            raise RuntimeError(
                "set_prev_frame() 仅在 encoding='framediff' 模式下可用")
        self.frame_diff_encoder.set_prev_frame(frame)

    def get_action(self, x: torch.Tensor) -> Union[int, torch.Tensor]:
        """
        推理便捷方法：返回动作 (离散) 或控制量 (回归)。

        Args:
            x: [1, C, H, W] 单张图像

        Returns:
            discrete:   action_id (int)
            regression: control tensor [control_dim]
        """
        with torch.no_grad():
            out = self.step(x)
            if self.head_type == 'discrete':
                return out.argmax(dim=1).item()
            else:
                return out.squeeze(0)


# ============================================================================
# 工厂函数
# ============================================================================
def build_corridor_net(
    head_type: str = 'discrete',
    num_actions: int = 3,
    control_dim: int = 1,
    encoding: str = 'rate',
    T: int = 8,
    neuron_type: str = 'APLIF',
    residual_mode: str = 'ADD',
    raw_in_channels: int = 3,
    use_tanh: bool = True,
    v_max: float = 0.22,
    w_max: float = 2.84,
    dropout: float = 0.3,
    pretrained_backbone: Optional[str] = None,
) -> CorridorPolicyNet:
    """
    构建走廊导航策略网络的便捷工厂函数。

    Args:
        head_type: 'discrete' 或 'regression'
        num_actions: 离散动作数 (3 或 5)
        control_dim: 回归维度 (1=angular_z, 2=[linear_x, angular_z])
        encoding: 'rate' 或 'framediff'
        T: 时间步 (rate 模式推荐 8; framediff 模式推荐 1)
        neuron_type: 'LIF'/'PLIF'/'ALIF'/'APLIF'
        residual_mode: 'standard'/'ADD'
        raw_in_channels: 原始图像通道数 (通常 3=RGB)
        use_tanh: 回归是否用 tanh + scale
        v_max: TurtleBot 最大线速度 (m/s)
        w_max: TurtleBot 最大角速度 (rad/s)
        dropout: head 的 dropout
        pretrained_backbone: 预训练 backbone 的 checkpoint 路径
            会自动加载 model_state_dict 中 backbone 相关的权重

    Returns:
        CorridorPolicyNet 实例
    """
    # 确定 backbone 的输入通道
    if encoding == 'framediff':
        backbone_in_channels = raw_in_channels * 2  # 6 通道
        if T != 1:
            import warnings
            warnings.warn(
                f"framediff 模式推荐 T=1 (当前 T={T})，"
                f"因为帧差编码已经隐含了时间信息。")
    else:
        backbone_in_channels = raw_in_channels

    # 构建 backbone (用足够大的 num_classes 当占位，实际用 features 不用 fc)
    backbone = SorResNet(
        block=BasicBlock,
        layers=[18, 18, 18],  # ResNet-110 结构
        num_classes=num_actions if head_type == 'discrete' else control_dim,
        T=T,
        neuron_type=neuron_type,
        residual_mode=residual_mode,
        in_channels=backbone_in_channels,
    )

    # 加载预训练权重 (可选)
    if pretrained_backbone is not None:
        ckpt = torch.load(pretrained_backbone, map_location='cpu',
                          weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt

        # 过滤不匹配的键 (conv1 通道变了、fc 维度变了)
        model_state = backbone.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        backbone.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[CorridorPolicyNet] 跳过不匹配的权重: {skipped}")
        print(f"[CorridorPolicyNet] 加载了 {len(filtered)}/{len(state)} "
              f"个预训练权重")

    # 构建策略网络
    net = CorridorPolicyNet(
        backbone=backbone,
        head_type=head_type,
        num_actions=num_actions,
        control_dim=control_dim,
        encoding=encoding,
        raw_in_channels=raw_in_channels,
        use_tanh=use_tanh,
        v_max=v_max,
        w_max=w_max,
        dropout=dropout,
    )

    return net


# ============================================================================
# 快速测试
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  CorridorPolicyNet 快速测试")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Test 1: Rate + Discrete ---
    print("\n[Test 1] Rate + Discrete (3类, T=4)")
    net1 = build_corridor_net(
        head_type='discrete', num_actions=3,
        encoding='rate', T=4, neuron_type='APLIF')
    net1.to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    out = net1(x)
    print(f"  input: {x.shape} → output: {out.shape}")
    assert out.shape == (2, 3), f"期望 (2,3), 得到 {out.shape}"
    net1.reset_state()

    # --- Test 2: FrameDiff + Regression ---
    print("\n[Test 2] FrameDiff + Regression (dim=2, T=1)")
    net2 = build_corridor_net(
        head_type='regression', control_dim=2,
        encoding='framediff', T=1, neuron_type='APLIF')
    net2.to(device)
    x1 = torch.randn(2, 3, 32, 32).to(device)
    x2 = torch.randn(2, 3, 32, 32).to(device)
    net2.reset_state()
    out1 = net2.step(x1)  # 首帧，差分为 0
    out2 = net2.step(x2)  # 第二帧，有有效差分
    print(f"  frame1 → {out1.shape}, frame2 → {out2.shape}")
    assert out2.shape == (2, 2), f"期望 (2,2), 得到 {out2.shape}"

    # --- Test 3: 序列推理 ---
    print("\n[Test 3] 序列推理 (Rate, L=4)")
    net3 = build_corridor_net(
        head_type='discrete', num_actions=5,
        encoding='rate', T=4, neuron_type='LIF')
    net3.to(device)
    xs = torch.randn(2, 4, 3, 32, 32).to(device)
    outs = net3.forward_seq(xs)
    print(f"  input: {xs.shape} → output: {outs.shape}")
    assert outs.shape == (2, 4, 5), f"期望 (2,4,5), 得到 {outs.shape}"
    net3.reset_state()

    # --- Test 4: get_action ---
    print("\n[Test 4] get_action (在线推理)")
    net4 = build_corridor_net(
        head_type='discrete', num_actions=3,
        encoding='rate', T=4)
    net4.to(device).eval()
    frame = torch.randn(1, 3, 32, 32).to(device)
    action = net4.get_action(frame)
    print(f"  action_id = {action}")
    assert isinstance(action, int)

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
