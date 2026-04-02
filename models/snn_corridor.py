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

    def __init__(self, in_channels: int = 3, gain: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2
        self.gain = gain
        self.register_buffer('prev_frame', None)

    def set_prev_frame(self, frame: torch.Tensor):
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        self.prev_frame = frame.detach().clone()

    def reset(self):
        self.prev_frame = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prev_frame is None:
            diff_pos = torch.zeros_like(x)
            diff_neg = torch.zeros_like(x)
        else:
            prev = self.prev_frame
            if prev.shape[0] != x.shape[0]:
                prev = prev[:x.shape[0]]
            diff_pos = F.relu(x - prev)
            diff_neg = F.relu(prev - x)

        self.prev_frame = x.detach().clone()

        out = torch.cat([diff_pos, diff_neg], dim=1)
        if self.gain != 1.0:
            out = out * self.gain
        return out


# ============================================================================
# 输出头
# ============================================================================
class DiscreteHead(nn.Module):
    def __init__(self, feat_dim: int, num_actions: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_actions, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class RegressionHead(nn.Module):
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
        out = self.head(features)
        if self.use_tanh:
            out = torch.tanh(out)
            if self.control_dim == 1:
                out = out * self.w_max
            else:
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
        **kwargs,
    ):
        super().__init__()
        assert head_type in ('discrete', 'regression')
        assert encoding in ('rate', 'framediff')

        self.head_type = head_type
        self.encoding = encoding
        self.raw_in_channels = raw_in_channels
        self.kwargs = kwargs

        self.frame_diff_encoder = None
        if encoding == 'framediff':
            self.frame_diff_encoder = FrameDiffEncoder(
                raw_in_channels, gain=kwargs.get('framediff_gain', 1.0)
            )

        self.backbone = backbone
        self.feat_dim = 128 * BasicBlock.expansion

        if head_type == 'discrete':
            self.head = DiscreteHead(self.feat_dim, num_actions, dropout)
            self.num_actions = num_actions
        else:
            self.head = RegressionHead(
                self.feat_dim, control_dim, use_tanh, v_max, w_max, dropout
            )
            self.control_dim = control_dim

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding == 'framediff':
            return self.frame_diff_encoder(x)
        return x

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.backbone(x, return_features=True)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(x)
        features = self._extract_features(encoded)
        return self.head(features)

    def forward_seq(self, xs: torch.Tensor,
                    reset_at_start: bool = True) -> torch.Tensor:
        B, L, C, H, W = xs.shape

        if reset_at_start:
            self.reset_state()

        outputs = []
        for t in range(L):
            frame = xs[:, t]
            out = self.forward(frame)
            outputs.append(out)

            # rate 模式：每帧内部已完成 T 步展开，因此帧间 reset backbone
            if self.encoding == 'rate':
                functional.reset_net(self.backbone)

            # framediff 模式：保持状态跨帧累积

        return torch.stack(outputs, dim=1)

    def step(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        out = self.forward(frame)

        if self.encoding == 'rate':
            functional.reset_net(self.backbone)

        return out

    def reset_state(self):
        functional.reset_net(self)
        if self.frame_diff_encoder is not None:
            self.frame_diff_encoder.reset()

    def set_prev_frame(self, frame: torch.Tensor):
        if self.frame_diff_encoder is None:
            raise RuntimeError(
                "set_prev_frame() 仅在 encoding='framediff' 模式下可用"
            )
        self.frame_diff_encoder.set_prev_frame(frame)

    def get_action(self, x: torch.Tensor) -> Union[int, torch.Tensor]:
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
    framediff_gain: float = 1.0,

    # ===== 新增：神经元超参数 =====
    tau: float = 2.0,
    init_tau: float = 2.0,
    tau_adp: float = 20.0,
    init_tau_adp: float = 20.0,
    beta: float = 1.8,
    learn_tau_adp: bool = False,
    learn_beta: bool = False,
    use_extra_exp_leak: bool = False,
    extra_exp_leak_scale: float = 0.0,
) -> CorridorPolicyNet:
    """
    构建走廊导航策略网络。

    新增支持：
      - APLIF / ALIF 的阈值自适应超参数透传
      - LIF / PLIF / ALIF / APLIF 统一接口
    """
    if encoding == 'framediff':
        backbone_in_channels = raw_in_channels * 2
    else:
        backbone_in_channels = raw_in_channels

    # ===== 新增：统一传给 SorResNet -> build_neuron =====
    neuron_kwargs = dict(
        tau=tau,
        init_tau=init_tau,
        tau_adp=tau_adp,
        init_tau_adp=init_tau_adp,
        beta=beta,
        learn_tau_adp=learn_tau_adp,
        learn_beta=learn_beta,
        use_extra_exp_leak=use_extra_exp_leak,
        extra_exp_leak_scale=extra_exp_leak_scale,
    )

    backbone = SorResNet(
        block=BasicBlock,
        layers=[18, 18, 18],
        num_classes=num_actions if head_type == 'discrete' else control_dim,
        T=T,
        neuron_type=neuron_type,
        residual_mode=residual_mode,
        in_channels=backbone_in_channels,
        neuron_kwargs=neuron_kwargs,   # ===== 关键新增 =====
    )

    if pretrained_backbone is not None:
        ckpt = torch.load(pretrained_backbone, map_location='cpu',
                          weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt

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
        print(f"[CorridorPolicyNet] 加载了 {len(filtered)}/{len(state)} 个预训练权重")

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
        framediff_gain=framediff_gain,
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

    print("\n[Test 1] Rate + Discrete (3类, T=4, APLIF)")
    net1 = build_corridor_net(
        head_type='discrete',
        num_actions=3,
        encoding='rate',
        T=4,
        neuron_type='APLIF',
        init_tau=2.0,
        init_tau_adp=20.0,
        beta=1.8,
    )
    net1.to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    out = net1(x)
    print(f"  input: {x.shape} -> output: {out.shape}")
    assert out.shape == (2, 3)
    net1.reset_state()

    print("\n[Test 2] FrameDiff + Regression (dim=2, T=1, APLIF)")
    net2 = build_corridor_net(
        head_type='regression',
        control_dim=2,
        encoding='framediff',
        T=1,
        neuron_type='APLIF',
        init_tau=2.0,
        init_tau_adp=20.0,
        beta=1.8,
        framediff_gain=5.0,
    )
    net2.to(device)
    x1 = torch.randn(2, 3, 32, 32).to(device)
    x2 = torch.randn(2, 3, 32, 32).to(device)
    net2.reset_state()
    out1 = net2.step(x1)
    out2 = net2.step(x2)
    print(f"  frame1 -> {out1.shape}, frame2 -> {out2.shape}")
    assert out2.shape == (2, 2)

    print("\n[Test 3] 序列推理 (Rate, L=4)")
    net3 = build_corridor_net(
        head_type='discrete',
        num_actions=5,
        encoding='rate',
        T=4,
        neuron_type='ALIF',
        tau=2.0,
        tau_adp=20.0,
        beta=1.8,
    )
    net3.to(device)
    xs = torch.randn(2, 4, 3, 32, 32).to(device)
    outs = net3.forward_seq(xs)
    print(f"  input: {xs.shape} -> output: {outs.shape}")
    assert outs.shape == (2, 4, 5)
    net3.reset_state()

    print("\n[Test 4] get_action")
    net4 = build_corridor_net(
        head_type='discrete',
        num_actions=3,
        encoding='rate',
        T=4,
        neuron_type='APLIF',
        init_tau=2.0,
        init_tau_adp=20.0,
        beta=1.8,
    )
    net4.to(device).eval()
    frame = torch.randn(1, 3, 32, 32).to(device)
    action = net4.get_action(frame)
    print(f"  action_id = {action}")
    assert isinstance(action, int)

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)