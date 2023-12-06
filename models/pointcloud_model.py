# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import (OpenGLPerspectiveCameras,
                                PointsRasterizationSettings,
                                AlphaCompositor)
from pytorch3d.structures import Pointclouds
from scipy.spatial.transform import Rotation

from helpers import my_torch_helpers
from renderer.SpherePointsRasterizer import SpherePointsRasterizer
from renderer.SpherePointsRenderer import SpherePointsRenderer


class PointcloudModel(nn.Module):
  def __init__(self):
    super().__init__()

  def make_point_cloud(self,
                       depths,
                       images,
                       rots=None,
                       trans=None,
                       inv_rot_trans=True,
                       linearize_angle=np.deg2rad(10)):
    """Creates a single point cloud from the rendered depths.

    Args:
      depths: Depths.
      images: Panos.
      rots: Rotations to the reference.
      trans: Translations to the reference.
      inv_rot_trans: Should we invert the rotation and translations

    Returns:
      Vertices and colors of the point cloud.

    """

    # height, width = depths.shape[1:3]
    batch_size, height, width, channels = images.shape
    theta, phi = np.meshgrid(
      (np.arange(width) + 0.5) * (2 * np.pi / width) + np.pi / 2,
      (np.arange(height) + 0.5) * (np.pi / height))
    theta = torch.tensor(theta, dtype=depths.dtype, device=depths.device)
    phi = torch.tensor(phi, dtype=depths.dtype, device=depths.device)

    all_xyz_coords = []
    all_point_colors = []
    for batch in range(depths.shape[0]):

      xyz = my_torch_helpers.spherical_to_cartesian(
        theta, phi, depths[batch, :, :, 0])
      colors = images[batch, :, :, :].reshape((-1, channels))
      if rots is not None and trans is not None:
        if inv_rot_trans:
          m_rot_inv = rots[batch, :, :].detach().cpu().numpy()
          m_rot_inv = Rotation.from_matrix(m_rot_inv).inv().as_matrix()
          m_rot_inv = torch.tensor(m_rot_inv,
                                   device=rots.device,
                                   dtype=rots.dtype)
          m_trans = trans[batch, :]

          # Perform xyz = m_rot_inv @ xyz - m_trans
          xyz = torch.tensordot(m_rot_inv,
                                xyz.reshape((-1, 3)),
                                dims=([1], [1])).permute((1, 0))
          xyz = xyz - m_trans
        else:
          m_rot_inv = rots[batch, :, :]
          m_trans = trans[batch, :]
          # Perform xyz = m_rot_inv @ xyz + m_trans
          xyz = torch.tensordot(m_rot_inv,
                                xyz.reshape((-1, 3)),
                                dims=([1], [1])).permute((1, 0))
          xyz = xyz + m_trans

      xyz = xyz.reshape((-1, 3))
      all_xyz_coords.append(xyz)
      all_point_colors.append(colors)
    return Pointclouds(points=all_xyz_coords, features=all_point_colors)

  def render_point_cloud(self, pointcloud, rot=None, trans=None,
                         size=256, radius=0.01, points_per_pixel=8,
                         linearize_angle=np.deg2rad(10)):
    """Renders a point cloud.

    Args:
      pointcloud: point cloud.
      rot: rotation of points.
      trans: translation of points.

    Returns:
      Rendered images as a channels-last tensor.

    """
    # Prepares the renderer.
    if rot is None:
      rot = torch.eye(3, dtype=torch.float32)
    if trans is None:
      trans = torch.zeros(3, dtype=torch.float32)
    cameras = OpenGLPerspectiveCameras(device=pointcloud.device,
                                       R=rot,
                                       T=trans,
                                       fov=99999,
                                       znear=0.0001,
                                       zfar=100.0)
    raster_settings = PointsRasterizationSettings(
      image_size=size,
      radius=radius,
      points_per_pixel=points_per_pixel)
    rasterizer = SpherePointsRasterizer(cameras=cameras,
                                        raster_settings=raster_settings,
                                        linearize_angle=linearize_angle)
    renderer = SpherePointsRenderer(rasterizer=rasterizer,
                                    compositor=AlphaCompositor())
    return renderer(pointcloud)
