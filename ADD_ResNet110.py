import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from spikingjelly.clock_driven import neuron, layer
from spikingjelly.clock_driven import surrogate
from neuron_model import APLIFNode, build_neuron
from spikingjelly.clock_driven.neuron import ParametricLIFNode


__all__ = ['SorResNet', 'resnet44', 'resnet50', 'resnet56', 'resnet110', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']





def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        neuron_builder: Optional[Callable] = None,
        residual_mode: str = 'ADD'
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.residual_mode = residual_mode
        
        # 默认兼容：如果没有提供 neuron_builder，使用原始 APLIFNode
        if neuron_builder is None:
            neuron_builder = lambda: APLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn1 = neuron_builder()
        self.sn2 = neuron_builder()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity
            out = self.sn2(out)
        else:
            if self.residual_mode == 'ADD':
                # ADD 残差：out + identity - 2*out*identity (XOR-like)
                out1 = out + identity - 2 * (out * identity)
                out = self.conv2(out1)
                out = self.bn2(out)
                out = self.sn2(out)
                out = identity + out - (identity * out)  # OR-like 合并
            else:
                # standard 残差：经典 ResNet add
                out = self.conv2(out)
                out = self.bn2(out)
                out = out + identity
                out = self.sn2(out)

        return out


#----------------------------------------------------------------------------
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        neuron_builder: Optional[Callable] = None,
        residual_mode: str = 'ADD'
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.residual_mode = residual_mode
        
        # 默认兼容
        if neuron_builder is None:
            neuron_builder = lambda: APLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn1 = neuron_builder()
        self.sn2 = neuron_builder()
        self.sn3 = neuron_builder()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            out += identity
            out = self.sn3(out)
        else:
            if self.residual_mode == 'ADD':
                # ADD 残差逻辑
                out = self.sn3(out)
                out = identity + out - (identity * out)
            else:
                # standard 残差
                out = out + identity
                out = self.sn3(out)

        return out


class SorResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        return_features: bool = False,
        T: int = 8,
        neuron_type: str = 'APLIF',
        residual_mode: str = 'ADD',
        in_channels: int = 3
    ) -> None:
        super(SorResNet, self).__init__()
        self.return_features = return_features  # 控制是否返回中间特征
        self.T = T
        self.neuron_type = neuron_type
        self.residual_mode = residual_mode
        
        # 构建神经元工厂
        self._neuron_builder = build_neuron(neuron_type)
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn = self._neuron_builder()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(128 * block.expansion, num_classes, bias=False),
            # APLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            neuron_builder=self._neuron_builder,
                            residual_mode=self.residual_mode))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                neuron_builder=self._neuron_builder,
                                residual_mode=self.residual_mode))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, return_features: bool = None) -> Union[Tensor, tuple]:
        """
        Args:
            x: 输入图像 tensor
            return_features: 是否返回特征向量 (覆盖实例属性)
        
        Returns:
            如果 return_features=False: 返回分类 logits
            如果 return_features=True: 返回 (logits, features)
                features 为 avgpool 之后的高维特征向量 (用于导航决策)
        """
        # See note [TorchScript super()]
        if return_features is None:
            return_features = self.return_features
            
# -------------------------------------------------- 静态数据集
        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.sn(out)
        out1 = self.layer1(out1)
        out1 = self.layer2(out1)
        out_spike = self.layer3(out1)

        # 时间步展开: 对静态图像进行多步脉冲编码
        for t in range(1, self.T):
            out2 = self.sn(out)
            out2 = self.layer1(out2)
            out2 = self.layer2(out2)
            out_spike += self.layer3(out2)
        out = out_spike / self.T
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        # 高维视觉特征 (用于导航决策模块)
        features = out
        
        # 分类头
        out = self.fc(out)
        
        if return_features:
            return out, features
        else:
            return out

# --------------------------------------神经形态数据集
#         x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W] 神经形态数据集
#         out = self.conv1(x[0])
#         out = self.bn1(out)
#         out = self.sn(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out_spikes = self.fc(out)
#         for t in range(1, x.shape[0]):
#             out = self.conv1(x[t])
#             out = self.bn1(out)
#             out = self.sn(out)
#             out = self.layer1(out)
#             out = self.layer2(out)
#             out = self.layer3(out)
#             out = self.avgpool(out)
#             out = torch.flatten(out, 1)
#             out_spikes += self.fc(out)
#         return out_spikes / x.shape[0]
# ----------------------------------------------------

    def forward(self, x: Tensor, return_features: bool = None) -> Union[Tensor, tuple]:
        return self._forward_impl(x, return_features)
    
    def extract_features(self, x: Tensor) -> Tensor:
        """
        专门用于提取高维视觉特征的接口 (用于导航决策)
        
        Args:
            x: 输入图像 tensor
        
        Returns:
            features: 高维特征向量 (batch_size, feature_dim)
        """
        _, features = self._forward_impl(x, return_features=True)
        return features


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> SorResNet:
    model = SorResNet(block, layers, **kwargs)
    return model




def resnet44(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SorResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet44', Bottleneck, [3, 8, 3], pretrained, progress,
                   **kwargs)



def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SorResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 10, 3], pretrained, progress,
                   **kwargs)


def resnet56(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SorResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet56', Bottleneck, [3, 12, 3], pretrained, progress,
                   **kwargs)



def resnet110(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SorResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet110', Bottleneck, [19, 14, 3], pretrained, progress,
                   **kwargs)




def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SorResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SorResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SorResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SorResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SorResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SorResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
