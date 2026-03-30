"""
统一的走廊导航三模块推理封装（离线/在线共享，不依赖 ROS2）
===========================================================

本文件提供三个可复用推理类：
  1) JunctionLRInfer   : 路口左右二分类
  2) Stage3Infer       : 三阶段分类（Approach/Turn/Recover）
  3) StraightKeepInfer : 直行纠偏回归（输出 omega）

设计目标：
  - 统一 checkpoint 加载方式（优先读取 ckpt['config']）
  - 统一单帧输入接口（numpy / PIL / torch.Tensor）
  - 自动完成 resize + normalize + tensor 化
  - 统一结构化 dict 输出，便于后续 replay 脚本和 ROS2 节点复用

示例命令：
  # stage3（APLIF_ADD_T4 主线）
  python inference/corridor_module_infer.py ^
      --module stage3 ^
      --ckpt checkpoint/corridor_task/corridor_task_stage3_APLIF_ADD_T4/best_model.ckpt ^
      --image demo.jpg ^
      --device cuda:0

  # junction_lr（APLIF_ADD_T4 主线）
  python inference/corridor_module_infer.py ^
      --module junction_lr ^
      --ckpt checkpoint/corridor_task/corridor_task_junction_lr_APLIF_ADD_T4/best_model.ckpt ^
      --image demo.jpg

  # straight_keep（APLIF_ADD_T4_seq4 主线，单帧在线调用）
  python inference/corridor_module_infer.py ^
      --module straight_keep ^
      --ckpt checkpoint/corridor/straight_keep_reg_APLIF_ADD_T4_seq4/best_model.ckpt ^
      --image demo.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# 允许以单文件脚本方式运行时，找到仓库根目录下的 models 包
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.snn_corridor import build_corridor_net  # noqa: E402


ImageInput = Union[np.ndarray, Image.Image, torch.Tensor]


def _as_int(value: Any, default: int) -> int:
    """安全转 int，失败时回退 default。"""
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_str(value: Any, default: str) -> str:
    """安全转 str，空值时回退 default。"""
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    从不同 checkpoint 格式中提取 state_dict。
    兼容:
      - {'model_state_dict': ...}
      - {'state_dict': ...}
      - 直接就是 state_dict
    """
    if isinstance(ckpt_obj, dict):
        for key in ('model_state_dict', 'state_dict', 'model', 'net'):
            state = ckpt_obj.get(key, None)
            if isinstance(state, dict):
                return state
        # 可能本身就是 state_dict（值应主要是 Tensor）
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            tensor_like = sum(1 for v in ckpt_obj.values() if torch.is_tensor(v))
            if tensor_like > 0 and tensor_like >= max(1, len(ckpt_obj) // 2):
                return ckpt_obj
    raise RuntimeError('无法从 checkpoint 提取 state_dict')


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """去掉 DDP 常见的 `module.` 前缀。"""
    if not state_dict:
        return state_dict
    if all(k.startswith('module.') for k in state_dict.keys()):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


class _BaseCorridorInfer:
    """
    统一基础推理封装。

    说明：
      - return dict 的具体结构由子类定义
      - 仅做推理，不涉及训练逻辑
    """

    def __init__(
        self,
        ckpt_path: str,
        *,
        device: Optional[str] = None,
        head_type: str = 'discrete',
        default_num_actions: int = 3,
        default_control_dim: int = 1,
        default_encoding: str = 'rate',
        default_T: int = 4,
        default_neuron_type: str = 'APLIF',
        default_residual_mode: str = 'ADD',
        default_img_h: int = 48,
        default_img_w: int = 64,
    ) -> None:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'checkpoint 不存在: {ckpt_path}')

        self.ckpt_path = ckpt_path
        self.device = torch.device(
            device if device is not None
            else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        self.head_type = head_type

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.ckpt = ckpt
        self.cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}

        # 优先使用 config；缺失时使用稳健 fallback
        self.encoding = _as_str(self.cfg.get('encoding'), default_encoding)
        self.T = _as_int(self.cfg.get('T'), default_T)
        self.neuron_type = _as_str(self.cfg.get('neuron_type'), default_neuron_type)
        self.residual_mode = _as_str(self.cfg.get('residual_mode'), default_residual_mode)
        self.img_h = _as_int(self.cfg.get('img_h'), default_img_h)
        self.img_w = _as_int(self.cfg.get('img_w'), default_img_w)

        self.num_actions = _as_int(
            self.cfg.get('task_num_classes', self.cfg.get('num_classes')),
            default_num_actions
        )
        self.control_dim = _as_int(self.cfg.get('control_dim'), default_control_dim)
        self.seq_len = _as_int(self.cfg.get('seq_len'), 1)
        self.is_sequence = bool(self.cfg.get('is_sequence', False))

        self.net = build_corridor_net(
            head_type=self.head_type,
            num_actions=self.num_actions,
            control_dim=self.control_dim,
            encoding=self.encoding,
            T=self.T,
            neuron_type=self.neuron_type,
            residual_mode=self.residual_mode,
            raw_in_channels=3,
            # 与训练链一致：回归默认允许 tanh 限幅；分类无影响
            use_tanh=(self.head_type == 'regression'),
        )

        state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
        try:
            self.net.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # 兼容部分历史 checkpoint 键不完全对齐场景
            self.net.load_state_dict(state_dict, strict=False)

        self.net.to(self.device)
        self.net.eval()

    def reset_state(self) -> None:
        """
        重置模型内部脉冲状态。
        在线串流推理切换 run 时建议调用一次。
        """
        if hasattr(self.net, 'reset_state'):
            self.net.reset_state()

    def _to_chw_tensor_01(self, image: ImageInput) -> torch.Tensor:
        """
        将输入统一为 CHW, float32, [0,1]。
        支持:
          - PIL.Image
          - numpy.ndarray (HWC 或 CHW)
          - torch.Tensor  (HWC / CHW / 1xCHW)
        """
        if isinstance(image, Image.Image):
            img = image.convert('RGB')
            x = TF.pil_to_tensor(img).float() / 255.0  # [3,H,W]

        elif isinstance(image, np.ndarray):
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.ndim != 3:
                raise ValueError(f'numpy 输入维度必须是 2/3，当前: {arr.shape}')

            # 若是 CHW 则转为 HWC，再转 CHW，保证逻辑一致
            if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))

            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]

            x = torch.from_numpy(np.ascontiguousarray(arr)).float()
            if x.max().item() > 1.5:
                x = x / 255.0
            x = x.permute(2, 0, 1)  # HWC -> CHW

        elif torch.is_tensor(image):
            x = image.detach().cpu()
            if x.ndim == 4:
                if x.shape[0] != 1:
                    raise ValueError(f'tensor 输入若为 4 维需 batch=1，当前: {tuple(x.shape)}')
                x = x[0]
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.ndim != 3:
                raise ValueError(f'tensor 输入维度必须是 2/3/4，当前: {tuple(x.shape)}')

            # 支持 HWC / CHW
            if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
                pass
            elif x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1)
            else:
                raise ValueError(f'无法判断 tensor 通道维，shape={tuple(x.shape)}')

            x = x.float()
            vmax = float(x.max().item())
            vmin = float(x.min().item())
            if vmax > 1.5:
                x = x / 255.0
            elif vmin < 0.0 and vmax <= 1.0:
                # 常见 [-1,1] 输入，先回到 [0,1] 再按本模块标准归一化
                x = (x + 1.0) / 2.0

            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            elif x.shape[0] > 3:
                x = x[:3]

        else:
            raise TypeError(
                f'不支持的输入类型: {type(image)}。'
                f'请使用 numpy / PIL / torch.Tensor'
            )

        x = x.clamp(0.0, 1.0)
        return x

    def _preprocess(self, image: ImageInput) -> torch.Tensor:
        """
        输入预处理：
          1) 转 CHW float [0,1]
          2) Resize 到训练分辨率
          3) Normalize(mean=0.5,std=0.5)
          4) 补 batch 维并移动到 device
        """
        x = self._to_chw_tensor_01(image)
        x = TF.resize(x, [self.img_h, self.img_w], antialias=True)
        x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = x.unsqueeze(0).to(self.device, non_blocking=True)
        return x

    @torch.no_grad()
    def _forward_single(self, image: ImageInput) -> torch.Tensor:
        """单帧前向，返回 shape=[D] 的 CPU tensor。"""
        x = self._preprocess(image)
        if hasattr(self.net, 'step'):
            out = self.net.step(x)
        else:
            out = self.net(x)
        out = out.detach().float().cpu().squeeze(0)
        return out


class JunctionLRInfer(_BaseCorridorInfer):
    """
    路口左右判定推理封装。

    主线配置对齐：
      - APLIF_ADD_T4
    """

    LABELS: Sequence[str] = ('Left', 'Right')

    def __init__(self, ckpt_path: str, device: Optional[str] = None) -> None:
        super().__init__(
            ckpt_path,
            device=device,
            head_type='discrete',
            default_num_actions=2,
            default_control_dim=1,
            default_encoding='rate',
            default_T=4,
            default_neuron_type='APLIF',
            default_residual_mode='ADD',
            default_img_h=48,
            default_img_w=64,
        )

    @torch.no_grad()
    def predict(self, image: ImageInput) -> Dict[str, Any]:
        logits = self._forward_single(image)
        probs_t = torch.softmax(logits, dim=0)
        pred_id = int(torch.argmax(probs_t).item())
        labels = list(self.LABELS)
        if self.num_actions != len(labels):
            labels = [f'class_{i}' for i in range(self.num_actions)]

        probs = {
            labels[i]: float(probs_t[i].item())
            for i in range(min(len(labels), probs_t.numel()))
        }

        pred_label = labels[pred_id] if pred_id < len(labels) else str(pred_id)
        confidence = float(probs_t[pred_id].item())
        return {
            'pred_label': pred_label,
            'pred_id': pred_id,
            'probs': probs,
            'confidence': confidence,
        }

    __call__ = predict


class Stage3Infer(_BaseCorridorInfer):
    """
    stage3 三阶段推理封装。

    主线配置对齐：
      - APLIF_ADD_T4
    """

    LABELS: Sequence[str] = ('Approach', 'Turn', 'Recover')

    def __init__(self, ckpt_path: str, device: Optional[str] = None) -> None:
        super().__init__(
            ckpt_path,
            device=device,
            head_type='discrete',
            default_num_actions=3,
            default_control_dim=1,
            default_encoding='rate',
            default_T=4,
            default_neuron_type='APLIF',
            default_residual_mode='ADD',
            default_img_h=48,
            default_img_w=64,
        )

    @torch.no_grad()
    def predict(self, image: ImageInput) -> Dict[str, Any]:
        logits = self._forward_single(image)
        probs_t = torch.softmax(logits, dim=0)
        pred_id = int(torch.argmax(probs_t).item())
        labels = list(self.LABELS)
        if self.num_actions != len(labels):
            labels = [f'class_{i}' for i in range(self.num_actions)]

        probs = {
            labels[i]: float(probs_t[i].item())
            for i in range(min(len(labels), probs_t.numel()))
        }

        pred_stage = labels[pred_id] if pred_id < len(labels) else str(pred_id)
        confidence = float(probs_t[pred_id].item())
        return {
            'pred_stage': pred_stage,
            'pred_id': pred_id,
            'probs': probs,
            'confidence': confidence,
        }

    __call__ = predict


class StraightKeepInfer(_BaseCorridorInfer):
    """
    直行纠偏回归推理封装。

    主线配置对齐：
      - APLIF_ADD_T4_seq4
    说明：
      - 即使 seq4 训练，在线也可按单帧连续调用；如需切换 run，调用 reset_state()。
    """

    def __init__(self, ckpt_path: str, device: Optional[str] = None) -> None:
        super().__init__(
            ckpt_path,
            device=device,
            head_type='regression',
            default_num_actions=1,
            default_control_dim=1,
            default_encoding='rate',
            default_T=4,
            default_neuron_type='APLIF',
            default_residual_mode='ADD',
            default_img_h=48,
            default_img_w=64,
        )
        # 与 APLIF_ADD_T4_seq4 主线一致：配置缺失时给出 seq_len=4 语义
        if self.seq_len <= 1:
            self.seq_len = 4

    @torch.no_grad()
    def predict(self, image: ImageInput) -> Dict[str, Any]:
        out = self._forward_single(image)
        if out.ndim == 0:
            omega = float(out.item())
        elif out.numel() == 1:
            omega = float(out.reshape(-1)[0].item())
        else:
            # 若控制维度为 2，按训练定义 [v, w]，优先取角速度 w
            idx = 1 if out.numel() >= 2 else 0
            omega = float(out.reshape(-1)[idx].item())

        return {
            'omega_cmd_raw': float(omega),
            'omega_abs': float(abs(omega)),
        }

    __call__ = predict


def _build_infer(module_name: str, ckpt: str, device: Optional[str]) -> _BaseCorridorInfer:
    module_name = module_name.lower().strip()
    if module_name == 'junction_lr':
        return JunctionLRInfer(ckpt_path=ckpt, device=device)
    if module_name == 'stage3':
        return Stage3Infer(ckpt_path=ckpt, device=device)
    if module_name == 'straight_keep':
        return StraightKeepInfer(ckpt_path=ckpt, device=device)
    raise ValueError(f'未知 module: {module_name}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='统一三模块推理入口（离线/在线共享，不依赖 ROS2）'
    )
    parser.add_argument(
        '--module',
        type=str,
        required=True,
        choices=['junction_lr', 'stage3', 'straight_keep'],
        help='选择推理模块',
    )
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint 路径')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='推理设备，例如 cpu / cuda:0（默认自动选择）',
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert('RGB')
    infer = _build_infer(args.module, args.ckpt, args.device)
    result = infer.predict(image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
