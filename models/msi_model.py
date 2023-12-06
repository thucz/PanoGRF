

import os
import sys

import numpy as np
import torch
from pytorch3d.renderer import (OpenGLPerspectiveCameras,
                                PointsRasterizationSettings, AlphaCompositor)
from pytorch3d.structures import Pointclouds
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as F

from helpers import my_helpers
from helpers import my_torch_helpers
from models.common_blocks import (ConvBlock, Conv3DBlock, Conv3DBlockv2,
                                  ConvBlock2, UNet2)
from models.inpainting_unet import InpaintNet
from renderer.SpherePointsRasterizer import SpherePointsRasterizer
from renderer.SpherePointsRenderer import SpherePointsRenderer


class MSIModel(nn.Module):
  def __init__(self):
    super().__init__()

  def calculate_cost_volume_erp(self,
                                images,
                                depths,
                                trans_norm,
                                cost_type="abs_diff"):
    """Calculates a cost volume for ERP images via backwards warping.

    Panos should be moving forward between images 0 and 1.

    Args:
      images: Tensor of shape (B, 2, H, W, C).
        The target image should be in index 1 along dim 1.
      depths: Tensor of depths to test.
      trans_norm: Norm of the translation.
      cost_type: Type of the cost volume.
      direction: Direction of cost volume.

    Returns:
      Tensor of shape (B, L, H, W, C).
    """
    batch_size, image_ct, height, width, channels = images.shape
    other_image = images[:, 0]
    other_image_cf = other_image.permute((0, 3, 1, 2))
    reference_image_cf = images[:, 1].permute((0, 3, 1, 2))
    phi = torch.arange(0,
                       height,
                       device=images.device,
                       dtype=images.dtype)
    phi = (phi + 0.5) * (np.pi / height)
    theta = torch.arange(0,
                         width,
                         device=images.device,
                         dtype=images.dtype)
    theta = (theta + 0.5) * (2 * np.pi / width) + np.pi / 2
    phi, theta = torch.meshgrid(phi, theta)
    translation = torch.stack(
      (torch.zeros_like(trans_norm), torch.zeros_like(trans_norm), trans_norm),
      dim=1)
    xyz = my_torch_helpers.spherical_to_cartesian(theta, phi, r=1)
    xyz = xyz[None, :, :, :].expand(batch_size, height, width, 3)

    cost_volume = []
    for i, depth in enumerate(depths):
      m_xyz = depth * xyz - translation[:, None, None, :]
      uv = my_torch_helpers.cartesian_to_spherical(m_xyz)
      u = torch.fmod(uv[..., 0] - np.pi / 2 + 4 * np.pi, 2 * np.pi) / np.pi - 1
      v = 2 * (uv[..., 1] / np.pi) - 1
      cv_image = torch.nn.functional.grid_sample(
        other_image_cf,
        torch.stack((u, v,), dim=-1),
        mode='bilinear',
        align_corners=True)
      if cost_type == 'abs_diff':
        cv_image = torch.abs(cv_image - reference_image_cf)
      elif cost_type != 'none':
        raise ValueError('Unknown cost type')
      cost_volume.append(cv_image)

    cost_volume = torch.stack(cost_volume, dim=1).permute((0, 1, 3, 4, 2))
    return cost_volume