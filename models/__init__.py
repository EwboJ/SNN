"""
models 包
==========
SNN 走廊导航模型。

Usage:
    from models.snn_corridor import CorridorPolicyNet, build_corridor_net
"""

from .snn_corridor import (
    CorridorPolicyNet,
    FrameDiffEncoder,
    DiscreteHead,
    RegressionHead,
    build_corridor_net,
)

__all__ = [
    'CorridorPolicyNet',
    'FrameDiffEncoder',
    'DiscreteHead',
    'RegressionHead',
    'build_corridor_net',
]
