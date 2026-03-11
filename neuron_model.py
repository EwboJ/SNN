from abc import abstractmethod
from typing import Callable, overload
import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, base, lava_exchange
from spikingjelly import configure
import math
import numpy as np
import logging
from spikingjelly.clock_driven.neuron import BaseNode

try:
    import cupy
    from . import neuron_kernel, cu_kernel_opt
except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    cupy = None
    neuron_kernel = None
    cu_kernel_opt = None

try:
    import lava.lib.dl.slayer as slayer

except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    slayer = None

def check_backend(backend: str):
    if backend == 'torch':
        return
    elif backend == 'cupy':
        assert cupy is not None, 'CuPy is not installed! You can install it from "https://github.com/cupy/cupy".'
    elif backend == 'lava':
        assert slayer is not None, 'Lava-DL is not installed! You can install it from "https://github.com/lava-nc/lava-dl".'
    else:
        raise NotImplementedError(backend)



class ALIFNode(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, cupy_fp32_inference=False):
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.decay_input = decay_input



        if cupy_fp32_inference:
            check_backend('cupy')
        self.cupy_fp32_inference = cupy_fp32_inference

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.as_tensor(self.v, device=x.device)
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau - torch.exp(self.v - self.v_threshold) * 0.2 #* 9e-1
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - 1. / self.tau) + x
            else:
                self.v = self.v - (self.v - self.v_reset) / self.tau + x


    def forward(self, x: torch.Tensor):
        if self.cupy_fp32_inference and cupy is not None and not self.training and x.dtype == torch.float32:
            # cupy is installed && eval mode && fp32
            device_id = x.get_device()
            if device_id < 0:
                return super().forward(x)

            # use cupy to accelerate
            if isinstance(self.v, float):
                v = torch.zeros_like(x)
                if self.v != 0.:
                    torch.fill_(v, self.v)
                self.v = v

            if self.v_reset is None:
                hard_reset = False
            else:
                hard_reset = True

            code = rf'''
                extern "C" __global__
                void LIFNode_{'hard' if hard_reset else 'soft'}_reset_decayInput_{self.decay_input}_inference_forward(
                const float * x, const float & v_threshold, {'const float & v_reset,' if hard_reset else ''} const float & tau,
                float * spike, float * v,
                const int & numel)
            '''

            code += r'''
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < numel)
                    {

            '''

            if self.decay_input:
                if hard_reset:
                    code += r'''
                     v[index] += (x[index] - (v[index] - v_reset)) / tau;
                            '''
                else:
                    code += r'''
                     v[index] += (x[index] - v[index]) / tau;
                    '''
            else:
                if hard_reset:
                    code += r'''
                     v[index] = x[index] + v[index] - (v[index] - v_reset) / tau;
                            '''
                else:
                    code += r'''
                     v[index] = x[index] + v[index] * (1.0f - 1.0f / tau);
                    '''

            code += rf'''\
            spike[index] = (float) (v[index] >= v_threshold);{'v[index] = (1.0f - spike[index]) * v[index] + spike[index] * v_reset;' if hard_reset else 'v[index] -= spike[index] * v_threshold;'}
            '''
            code += r'''
                    }
                }
            '''
            if hasattr(self, 'cp_kernel'):
                if self.cp_kernel.code != code:
                    # replace codes
                    del self.cp_kernel
                    self.cp_kernel = cupy.RawKernel(code,f"LIFNode_{'hard' if hard_reset else 'soft'}_reset_decayInput_{self.decay_input}_inference_forward",options=configure.cuda_compiler_options,backend=configure.cuda_compiler_backend)
            else:
                self.cp_kernel = cupy.RawKernel(code,f"LIFNode_{'hard' if hard_reset else 'soft'}_reset_decayInput_{self.decay_input}_inference_forward",options=configure.cuda_compiler_options,backend=configure.cuda_compiler_backend)

            with cu_kernel_opt.DeviceEnvironment(device_id):
                numel = x.numel()
                threads = configure.cuda_threads
                blocks = cu_kernel_opt.cal_blocks(numel)
                cp_numel = cupy.asarray(numel)
                cp_v_threshold = cupy.asarray(self.v_threshold, dtype=np.float32)
                if hard_reset:
                    cp_v_reset = cupy.asarray(self.v_reset, dtype=np.float32)
                cp_tau = cupy.asarray(self.tau, dtype=np.float32)
                spike = torch.zeros_like(x)
                if hard_reset:
                    x, cp_v_threshold, cp_v_reset, cp_tau, spike, self.v, cp_numel = cu_kernel_opt.get_contiguous(x,cp_v_threshold,cp_v_reset,cp_tau,spike,self.v,cp_numel)
                    kernel_args = [x, cp_v_threshold, cp_v_reset, cp_tau, spike, self.v, cp_numel]
                else:
                    x, cp_v_threshold, cp_tau, spike, self.v, cp_numel = cu_kernel_opt.get_contiguous(x, cp_v_threshold,cp_tau, spike,self.v, cp_numel)
                    kernel_args = [x, cp_v_threshold, cp_tau, spike, self.v, cp_numel]

                self.cp_kernel(
                    (blocks,), (threads,),
                    cu_kernel_opt.wrap_args_to_raw_kernel(
                        device_id,
                        *kernel_args
                    )
                )
                return spike
        else:
            return super().forward(x)


#_______________________________________________________________________________________
# 自适应膜电位时间常数tau
class APLIFNode(BaseNode):
    def __init__(self, init_tau: float = 2.0, decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        """
        * :ref:`API in English <ParametricLIFNode.__init__-en>`

        .. _ParametricLIFNode.__init__-cn:

        :param init_tau: 膜电位时间常数的初始值
        :type init_tau: float

        :param decay_input: 输入是否会衰减
        :type decay_input: bool

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_
        提出的 Parametric Leaky Integrate-and-Fire (PLIF)神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        若 ``decay_input == True``:

            .. math::
                V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        若 ``decay_input == False``:

            .. math::
                V[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        其中 :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`，:math:`w` 是可学习的参数。

        * :ref:`中文API <ParametricLIFNode.__init__-cn>`

        .. _ParametricLIFNode.__init__-en:

        :param init_tau: the initial value of membrane time constant
        :type init_tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Parametric Leaky Integrate-and-Fire (PLIF) neuron, which is proposed by `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_ and can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        IF ``decay_input == True``:

            .. math::
                V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        IF ``decay_input == False``:

            .. math::
                V[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        where :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`, :math:`w` is a learnable parameter.
        """

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.decay_input = decay_input
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))


    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.as_tensor(self.v, device=x.device)
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid() - torch.exp(self.v-self.v_threshold) * (1./ self.w.sigmoid()) / 10 # * 1e-1  # 0.1, 0.01
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - self.w.sigmoid()) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * self.w.sigmoid() + x


# ============================================================================
# 神经元工厂函数
# ============================================================================
def build_neuron(neuron_type: str = 'APLIF', **kwargs):
    """
    神经元工厂函数：根据类型字符串返回一个无参构造器 (callable)。
    
    用法：
        neuron_builder = build_neuron('APLIF', init_tau=2.0)
        sn = neuron_builder()  # 创建一个 APLIFNode 实例
    
    Args:
        neuron_type: 神经元类型，支持 'LIF' / 'PLIF' / 'ALIF' / 'APLIF'
        **kwargs: 传递给神经元构造函数的额外参数
    
    Returns:
        callable: 无参构造器，每次调用返回一个新的神经元实例
    """
    from spikingjelly.clock_driven.neuron import LIFNode, ParametricLIFNode
    from spikingjelly.clock_driven import surrogate as _surrogate
    
    # 默认参数
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
                **kwargs
            )
    elif neuron_type == 'PLIF':
        init_tau = kwargs.pop('init_tau', 2.0)
        def _builder():
            return ParametricLIFNode(
                init_tau=init_tau,
                surrogate_function=default_surrogate,
                detach_reset=default_detach_reset,
                **kwargs
            )
    elif neuron_type == 'ALIF':
        tau = kwargs.pop('tau', 2.0)
        def _builder():
            return ALIFNode(
                tau=tau,
                surrogate_function=default_surrogate,
                detach_reset=default_detach_reset,
                **kwargs
            )
    elif neuron_type == 'APLIF':
        init_tau = kwargs.pop('init_tau', 2.0)
        def _builder():
            return APLIFNode(
                init_tau=init_tau,
                surrogate_function=default_surrogate,
                detach_reset=default_detach_reset,
                **kwargs
            )
    else:
        raise ValueError(
            f"未知的神经元类型: '{neuron_type}'。"
            f"支持的类型: LIF, PLIF, ALIF, APLIF"
        )
    
    return _builder
