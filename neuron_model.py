import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate
from spikingjelly.clock_driven.neuron import BaseNode


class ALIFNode(BaseNode):
    """
    真正的 ALIF:
    - 固定膜时间常数 tau
    - 自适应阈值: B_t = B0 + beta * a_t
    - a_t 随 spike 上升, 再按 tau_adp 衰减
    """
    def __init__(
        self,
        tau: float = 2.0,
        tau_adp: float = 20.0,
        beta: float = 1.8,
        decay_input: bool = True,
        v_threshold: float = 1.0,   # baseline threshold
        v_reset: float = 0.0,
        surrogate_function: Callable = surrogate.ATan(),
        detach_reset: bool = True,
        learn_tau_adp: bool = False,
        learn_beta: bool = False,
        use_extra_exp_leak: bool = False,
        extra_exp_leak_scale: float = 0.0,
    ):
        assert isinstance(tau, float) and tau > 1.0
        assert isinstance(tau_adp, float) and tau_adp > 1.0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

        self.tau = tau
        self.decay_input = decay_input
        self.v_threshold_base = float(v_threshold)

        # adaptation tau
        if learn_tau_adp:
            init_w_a = -math.log(tau_adp - 1.0)
            self.w_a = nn.Parameter(torch.tensor(init_w_a, dtype=torch.float32))
        else:
            self.register_buffer('tau_adp_buf', torch.tensor(float(tau_adp), dtype=torch.float32))

        # adaptation strength beta
        if learn_beta:
            beta_raw = math.log(math.exp(beta) - 1.0)
            self.beta_raw = nn.Parameter(torch.tensor(beta_raw, dtype=torch.float32))
        else:
            self.register_buffer('beta_buf', torch.tensor(float(beta), dtype=torch.float32))

        # 可选额外非线性漏电项，默认关闭
        self.use_extra_exp_leak = use_extra_exp_leak
        self.extra_exp_leak_scale = float(extra_exp_leak_scale)

        # 自适应阈值状态
        self.register_memory('a', 0.0)

    def membrane_alpha(self):
        return 1.0 / self.tau

    def adapt_alpha(self):
        if hasattr(self, 'w_a'):
            return self.w_a.sigmoid()   # in (0, 1)
        return 1.0 / self.tau_adp_buf

    def beta_value(self):
        if hasattr(self, 'beta_raw'):
            return F.softplus(self.beta_raw)
        return self.beta_buf

    def current_threshold(self):
        return self.v_threshold_base + self.beta_value() * self.a

    def neuronal_charge(self, x: torch.Tensor):
        if not isinstance(self.v, torch.Tensor):
            self.v = torch.zeros_like(x) + float(self.v)

        alpha_m = self.membrane_alpha()

        if self.decay_input:
            if self.v_reset is None:
                self.v = self.v + (x - self.v) * alpha_m
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * alpha_m
        else:
            if self.v_reset is None:
                self.v = self.v * (1.0 - alpha_m) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * alpha_m + x

        # 可选额外非线性漏电；不是 ALIF 核心
        if self.use_extra_exp_leak and self.extra_exp_leak_scale > 0:
            thr = self.current_threshold()
            self.v = self.v - torch.exp(self.v - thr) * self.extra_exp_leak_scale

    def forward(self, x: torch.Tensor):
        if not isinstance(self.v, torch.Tensor):
            self.v = torch.as_tensor(self.v, device=x.device)

        if not isinstance(self.a, torch.Tensor):
            self.a = torch.zeros_like(x)

        # 1) 膜电位积分
        self.neuronal_charge(x)

        # 2) 动态阈值
        thr = self.current_threshold()

        # 3) 发放
        spike = self.surrogate_function(self.v - thr)

        spike_for_reset = spike.detach() if self.detach_reset else spike

        # 4) reset
        if self.v_reset is None:
            # soft reset: 减去动态阈值
            self.v = self.v - spike_for_reset * thr
        else:
            # hard reset
            self.v = self.v * (1.0 - spike_for_reset) + self.v_reset * spike_for_reset

        # 5) 更新自适应阈值状态
        alpha_a = self.adapt_alpha()
        self.a = self.a + (spike_for_reset - self.a) * alpha_a

        return spike

    def extra_repr(self):
        with torch.no_grad():
            tau_a = 1.0 / self.adapt_alpha()
            beta = self.beta_value()
        return (
            super().extra_repr()
            + f', tau={self.tau:.4f}, tau_adp={tau_a.item():.4f}, beta={beta.item():.4f}'
        )


class APLIFNode(BaseNode):
    """
    真正的 APLIF:
    - PLIF: 可学习膜时间常数
    - ALIF: 自适应阈值
    """
    def __init__(
        self,
        init_tau: float = 2.0,         # membrane tau
        init_tau_adp: float = 20.0,    # adaptation tau
        beta: float = 1.8,             # threshold adaptation strength
        decay_input: bool = True,
        v_threshold: float = 1.0,      # baseline threshold
        v_reset: float = 0.0,
        surrogate_function: Callable = surrogate.ATan(),
        detach_reset: bool = True,
        learn_tau_adp: bool = False,
        learn_beta: bool = False,
        use_extra_exp_leak: bool = False,
        extra_exp_leak_scale: float = 0.0,
    ):
        assert isinstance(init_tau, float) and init_tau > 1.0
        assert isinstance(init_tau_adp, float) and init_tau_adp > 1.0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

        self.decay_input = decay_input
        self.v_threshold_base = float(v_threshold)

        # ---- PLIF: learnable membrane tau ----
        init_w_m = -math.log(init_tau - 1.0)
        self.w_m = nn.Parameter(torch.tensor(init_w_m, dtype=torch.float32))

        # ---- ALIF: adaptation tau ----
        if learn_tau_adp:
            init_w_a = -math.log(init_tau_adp - 1.0)
            self.w_a = nn.Parameter(torch.tensor(init_w_a, dtype=torch.float32))
        else:
            self.register_buffer('tau_adp_buf', torch.tensor(float(init_tau_adp), dtype=torch.float32))

        # ---- adaptation strength beta ----
        if learn_beta:
            beta_raw = math.log(math.exp(beta) - 1.0)
            self.beta_raw = nn.Parameter(torch.tensor(beta_raw, dtype=torch.float32))
        else:
            self.register_buffer('beta_buf', torch.tensor(float(beta), dtype=torch.float32))

        # 可选额外非线性漏电项，默认关闭
        self.use_extra_exp_leak = use_extra_exp_leak
        self.extra_exp_leak_scale = float(extra_exp_leak_scale)

        # adaptation trace
        self.register_memory('a', 0.0)

    def membrane_alpha(self):
        # alpha_m = 1 / tau_m = sigmoid(w_m)
        return self.w_m.sigmoid()

    def adapt_alpha(self):
        if hasattr(self, 'w_a'):
            return self.w_a.sigmoid()
        return 1.0 / self.tau_adp_buf

    def beta_value(self):
        if hasattr(self, 'beta_raw'):
            return F.softplus(self.beta_raw)
        return self.beta_buf

    def current_threshold(self):
        return self.v_threshold_base + self.beta_value() * self.a

    def neuronal_charge(self, x: torch.Tensor):
        if not isinstance(self.v, torch.Tensor):
            self.v = torch.zeros_like(x) + float(self.v)

        alpha_m = self.membrane_alpha()

        if self.decay_input:
            if self.v_reset is None:
                self.v = self.v + (x - self.v) * alpha_m
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * alpha_m
        else:
            if self.v_reset is None:
                self.v = self.v * (1.0 - alpha_m) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * alpha_m + x

        # 可选额外非线性漏电；不是 APLIF 核心
        if self.use_extra_exp_leak and self.extra_exp_leak_scale > 0:
            thr = self.current_threshold()
            self.v = self.v - torch.exp(self.v - thr) * self.extra_exp_leak_scale

    def forward(self, x: torch.Tensor):
        if not isinstance(self.v, torch.Tensor):
            self.v = torch.as_tensor(self.v, device=x.device)

        if not isinstance(self.a, torch.Tensor):
            self.a = torch.zeros_like(x)

        # 1) 膜电位积分
        self.neuronal_charge(x)

        # 2) 动态阈值
        thr = self.current_threshold()

        # 3) 发放
        spike = self.surrogate_function(self.v - thr)

        spike_for_reset = spike.detach() if self.detach_reset else spike

        # 4) reset
        if self.v_reset is None:
            self.v = self.v - spike_for_reset * thr
        else:
            self.v = self.v * (1.0 - spike_for_reset) + self.v_reset * spike_for_reset

        # 5) 更新自适应阈值状态
        alpha_a = self.adapt_alpha()
        self.a = self.a + (spike_for_reset - self.a) * alpha_a

        return spike

    def extra_repr(self):
        with torch.no_grad():
            tau_m = 1.0 / self.membrane_alpha()
            tau_a = 1.0 / self.adapt_alpha()
            beta = self.beta_value()
        return (
            super().extra_repr()
            + f', tau_m={tau_m.item():.4f}, tau_adp={tau_a.item():.4f}, beta={beta.item():.4f}'
        )


def build_neuron(neuron_type: str = 'APLIF', **kwargs):
    """
    返回一个无参构造器，每次调用生成一个新的 neuron instance
    """
    from spikingjelly.clock_driven.neuron import LIFNode, ParametricLIFNode
    from spikingjelly.clock_driven import surrogate as _surrogate

    default_surrogate = kwargs.pop('surrogate_function', _surrogate.ATan())
    default_detach_reset = kwargs.pop('detach_reset', True)

    neuron_type = neuron_type.upper()

    if neuron_type == 'LIF':
        tau = kwargs.pop('tau', 2.0)

        def _builder():
            return LIFNode(
                tau=tau,
                surrogate_function=default_surrogate,
                detach_reset=default_detach_reset,
                **kwargs,
            )

    elif neuron_type == 'PLIF':
        init_tau = kwargs.pop('init_tau', 2.0)

        def _builder():
            return ParametricLIFNode(
                init_tau=init_tau,
                surrogate_function=default_surrogate,
                detach_reset=default_detach_reset,
                **kwargs,
            )

    elif neuron_type == 'ALIF':
        tau = kwargs.pop('tau', 2.0)
        tau_adp = kwargs.pop('tau_adp', 20.0)
        beta = kwargs.pop('beta', 1.8)
        learn_tau_adp = kwargs.pop('learn_tau_adp', False)
        learn_beta = kwargs.pop('learn_beta', False)
        use_extra_exp_leak = kwargs.pop('use_extra_exp_leak', False)
        extra_exp_leak_scale = kwargs.pop('extra_exp_leak_scale', 0.0)

        def _builder():
            return ALIFNode(
                tau=tau,
                tau_adp=tau_adp,
                beta=beta,
                surrogate_function=default_surrogate,
                detach_reset=default_detach_reset,
                learn_tau_adp=learn_tau_adp,
                learn_beta=learn_beta,
                use_extra_exp_leak=use_extra_exp_leak,
                extra_exp_leak_scale=extra_exp_leak_scale,
                **kwargs,
            )

    elif neuron_type == 'APLIF':
        init_tau = kwargs.pop('init_tau', 2.0)
        init_tau_adp = kwargs.pop('init_tau_adp', 20.0)
        beta = kwargs.pop('beta', 1.8)
        learn_tau_adp = kwargs.pop('learn_tau_adp', False)
        learn_beta = kwargs.pop('learn_beta', False)
        use_extra_exp_leak = kwargs.pop('use_extra_exp_leak', False)
        extra_exp_leak_scale = kwargs.pop('extra_exp_leak_scale', 0.0)

        def _builder():
            return APLIFNode(
                init_tau=init_tau,
                init_tau_adp=init_tau_adp,
                beta=beta,
                surrogate_function=default_surrogate,
                detach_reset=default_detach_reset,
                learn_tau_adp=learn_tau_adp,
                learn_beta=learn_beta,
                use_extra_exp_leak=use_extra_exp_leak,
                extra_exp_leak_scale=extra_exp_leak_scale,
                **kwargs,
            )

    else:
        raise ValueError(
            f"未知的神经元类型: '{neuron_type}'。"
            f"支持的类型: LIF, PLIF, ALIF, APLIF"
        )

    return _builder