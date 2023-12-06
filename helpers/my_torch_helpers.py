# Lint as: python3
"""Helpers for pytorch.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt

from helpers import my_helpers


def spherical_to_cartesian(args, theta, phi, r=None):
  """Spherical to cartesian.

  We use the convention that y is up.
  Theta is the angle along the xz-plane from 0 to 2*pi.
  Phi is the angle from the positive-z axis from 0 to pi.

  Args:
    theta: value or array between 0 and 2*pi
    phi: value or array between 0 and pi
    r: None, value, or array of values >= 0 (Default value = None)

  Returns:
    array or nx3 array

  """
  if r is None:
    if torch.is_tensor(theta):
      r = torch.tensor(1, dtype=theta.dtype,
                       device=theta.device).expand_as(theta)
    else:
      r = 1
  if args["dataset_name"] == "m3d":
    tmp = r * torch.sin(phi)
    x = tmp * torch.cos(theta)
    y = r * torch.cos(phi)
    z = tmp * torch.sin(theta)
  elif args["dataset_name"] == "replica_test":
    # x = r * torch.sin(phi) * torch.cos(theta)
    # y = r * torch.sin(phi) * torch.sin(theta)
    # z = r * torch.cos(phi)
  
    x = r * torch.sin(theta) * torch.cos(phi)
    y = - r * torch.sin(phi)
    z = r * torch.cos(theta) * torch.cos(phi)
  elif args["dataset_name"] == "residential":
    # theta
    x = r * torch.cos(theta) * torch.cos(phi)
    z = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(phi)
  elif args["dataset_name"] in [ "CoffeeArea"]:
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)    
  else:
    raise Exception
  # import ipdb;ipdb.set_trace()
  return torch.stack((x, y, z), axis=-1)


def cartesian_to_spherical(args, xyz, eps=None, linearize_angle=np.deg2rad(10)):
  """Cartesian to Spherical coordinates.

  We use the convention that y is up.
  Theta is the angle along the xz-plane from 0 to 2*pi.
  Phi is the angle from the positive-z axis from 0 to pi.
  Returns in the order: theta, phi, radius stacked along the last axis.

  Args:
    xyz: nx3 array
    eps: Epsilon for safe_divide. Automatically chosen if none is provided.
    linearize_angle: Angle from the pole to linearize. Should be small.

  Returns:
    nx3 array

  """
  zero_tensor = torch.tensor(0, dtype=xyz.dtype, device=xyz.device)
  
  cos_deg = np.cos(linearize_angle)
  radius = torch.norm(xyz, dim=-1)
    
  x, y, z = torch.unbind(xyz, dim=-1)
  if args["dataset_name"] == "m3d":
    theta = torch.atan2(z, x)
    # y_over_r = safe_ops.safe_unsigned_div(y, radius, eps)
    y_over_r = torch.div(y, radius)
    y_over_r_valid = torch.lt(torch.abs(y_over_r), cos_deg)
    # Here we linearize acos near the poles.
    phi = torch.where(
      y_over_r_valid,
      torch.acos(torch.where(y_over_r_valid, y_over_r, zero_tensor)),
      torch.where(torch.ge(y, zero_tensor),
                  linearize_angle * (1 - y_over_r) / (1 - cos_deg),
                  np.pi - linearize_angle * (y_over_r + 1) / (-cos_deg + 1)))
  elif args["dataset_name"] == "replica_test":
    theta = torch.atan2(x, z)# 
    phi = -torch.asin(z/radius) #[-pi/2,+pi/2]
    # mask1 = theta.gt(np.pi)
    # theta[mask1] = theta[mask1] - 2*np.pi
    # mask2 = theta.lt(-1*np.pi)
    # theta[mask2] = theta[mask2] + 2*np.pi
  elif args["dataset_name"] == "residential":
    theta = -torch.atan2(-z, x)
    phi = torch.asin(y/radius)
    mask = torch.logical_and(
        theta.gt(np.pi*0.5), theta.le(2*np.pi))
    theta[mask] = theta[mask] - 2*np.pi
    # print("phi.max(), phi.min():", phi.max(), phi.min())
    # print("theta.max(), theta.min():", theta.max(), theta.min())
    # import ipdb;ipdb.set_trace()
  elif args["dataset_name"] in[ "CoffeeArea"]:
    theta = torch.atan2(y, x)
    phi = torch.acos(z/radius)
    mask1 = theta.lt(0)
    theta[mask1] = theta[mask1] + 2*np.pi
  else:
    raise Exception
  return torch.stack((theta, phi, radius), dim=-1)

def show_torch_image(image):
  """Show an image encoded as a pytorch tensor."""
  image_np = image[0, :, :, :].detach().cpu().numpy()
  image_np = np.clip(image_np, 0, 1)
  plt.imshow(image_np)


def save_torch_image(save_path, image, cmap="viridis", clamp=True,
                     size=None, vmin=None, vmax=None):
  """ Save an image to the path.

  Args:
    save_path: Path to save to.
    image: Tensor.
    cmap: Colormap for grayscale images.
    clamp: Clamp RGB to [0, 1].
    size: None.
    vmin: Min value.
    vmax: Max value.

  Returns:
    None

  """
  m_image = image
  if size is not None:
    m_image = resize_torch_images(image[:1], size=size)
  image_np = m_image[0].detach().cpu().numpy()
  if clamp:
    image_np = np.clip(image_np, 0, 1)
  plt.imsave(save_path, image_np, cmap=cmap, vmin=vmin, vmax=vmax)


def resize_torch_images(image, size, mode="bilinear", align_corners=None):
  """Resizes a channel-last tensor of images.

  Args:
    image: Image.
    size: New size.
    mode: Mode to do resizing.
    align_corners: Align corners.

  Returns:
    Resized image
  """
  image = image.permute((0, 3, 1, 2))
  if mode == "bilinear" and align_corners is None:
    align_corners = False
  image = torch.nn.functional.interpolate(image,
                                          size,
                                          mode=mode,
                                          align_corners=align_corners)
  return image.permute((0, 2, 3, 1))


def torch_image_to_np(image, size=None):
  if size is not None:
    image = resize_torch_images(image, size)
  image = image.detach().cpu().numpy()
  image = np.clip(image, 0, 1)
  return image


def sat_downsample(image, size):
  """Downsamples the image using a summed area table.

  Args:
    image: Image as tensor of shape BxHxWxC.
    size: new size (width, height)

  Returns:
    Downsampled image.

  """
  old_height, old_width = image.shape[1:3]
  new_width, new_height = size
  image = image.permute((0, 3, 1, 2))
  sat = torch.cumsum(image, 3)
  sat = torch.cumsum(sat, 2)
  sat_small = torch.nn.functional.interpolate(sat,
                                              size,
                                              mode="bilinear",
                                              align_corners=False)
  downsampled_image = sat_small[:, :, 1:, 1:] - sat_small[:, :, :-1, 1:] \
                      - sat_small[:, :, 1:, :-1] + sat_small[:, :, :-1, :-1]
  downsampled_image = downsampled_image / ((old_width / new_width) *
                                           (old_height / new_height))
  top_pixels = (sat_small[:, :, 0:1, 1:] - sat_small[:, :, 0:1, :-1]) \
               / (old_width / new_width)
  left_pixels = (sat_small[:, :, 1:, 0:1] - sat_small[:, :, :-1, 0:1]) \
                / (old_height / new_height)
  left_pixels = torch.cat((sat_small[:, :, 0:1, 0:1], left_pixels), dim=2)
  downsampled_image = torch.cat((top_pixels, downsampled_image), dim=2)
  downsampled_image = torch.cat((left_pixels, downsampled_image), dim=3)
  downsampled_image = downsampled_image.permute((0, 2, 3, 1))
  return downsampled_image


def masked_sat_downsample(image, size, valid_pixels):
  """Downsamples the image using a summed area table.

  Args:
    image: Image (channels last) BxHxWxC
    size: new size (width, height)
    valid pixels: Should be image of size BxHxWx1

  Returns:
    Downsampled image

  """
  old_height, old_width = image.shape[1:3]
  new_width, new_height = size
  image = image.permute((0, 3, 1, 2))
  sat = torch.cumsum(image, 3)
  sat = torch.cumsum(sat, 2)
  sat_small = torch.nn.functional.interpolate(sat,
                                              size,
                                              mode="bilinear",
                                              align_corners=False)
  downsampled_image = sat_small[:, :, 1:, 1:] - sat_small[:, :, :-1, 1:] \
                      - sat_small[:, :, 1:, :-1] + sat_small[:, :, :-1, :-1]
  top_pixels = (sat_small[:, :, 0:1, 1:] - sat_small[:, :, 0:1, :-1])
  left_pixels = (sat_small[:, :, 1:, 0:1] - sat_small[:, :, :-1, 0:1])
  left_pixels = torch.cat((sat_small[:, :, 0:1, 0:1], left_pixels), dim=2)
  downsampled_image = torch.cat((top_pixels, downsampled_image), dim=2)
  downsampled_image = torch.cat((left_pixels, downsampled_image), dim=3)
  downsampled_image = downsampled_image.permute((0, 2, 3, 1))

  mask_sat = valid_pixels.permute((0, 3, 1, 2))
  mask_sat = torch.cumsum(mask_sat, 3)
  mask_sat = torch.cumsum(mask_sat, 2)
  mask_sat = torch.nn.functional.interpolate(mask_sat,
                                             size,
                                             mode="bilinear",
                                             align_corners=False)
  downsampled_mask = mask_sat[:, :, 1:, 1:] - mask_sat[:, :, :-1, 1:] \
                     - mask_sat[:, :, 1:, :-1] + mask_sat[:, :, :-1, :-1]
  mask_sat = mask_sat
  top_pixels = (mask_sat[:, :, 0:1, 1:] - mask_sat[:, :, 0:1, :-1])
  left_pixels = (mask_sat[:, :, 1:, 0:1] - mask_sat[:, :, :-1, 0:1])
  left_pixels = torch.cat((mask_sat[:, :, 0:1, 0:1], left_pixels), dim=2)
  downsampled_mask = torch.cat((top_pixels, downsampled_mask), dim=2)
  downsampled_mask = torch.cat((left_pixels, downsampled_mask), dim=3)
  downsampled_mask = downsampled_mask.permute((0, 2, 3, 1))
  downsampled_mask = torch.where(
    torch.lt(torch.abs(downsampled_mask), 0.001),
    torch.tensor(1.0,
                 device=downsampled_mask.device,
                 dtype=downsampled_mask.dtype), downsampled_mask)
  return downsampled_image / downsampled_mask


def depth_to_turbo_colormap(depth, min_depth=None):
  """Non-differentiable depth to turbo colormap.

  Args:
    depth: Depth image of shape BxHxWxC.
    min_depth: Minimum depth.

  Returns:
    Depth image.

  """
  depth_np = depth.detach().cpu().numpy()
  colored_depth = my_helpers.depth_to_turbo_colormap(depth_np, min_depth)
  colored_depth = torch.tensor(colored_depth,
                               device=depth.device,
                               dtype=torch.float32)
  return colored_depth


def clamp_away_from(tensor, val=0, eps=1e-10):
  """Clamps all elements in tensor away from val.

  All values which are epsilon away stay the same.
  All values epsilon close are clamped to the nearest acceptable value.

  Args:
    tensor: Input tensor.
    val: Value you do not want in the tensor.
    eps: Distance away from the value which is acceptable.

  Returns:
    Input tensor where no elements are within eps of val.

  """
  if not torch.is_tensor(val):
    val = torch.tensor(val, device=tensor.device, dtype=tensor.dtype)
  tensor = torch.where(torch.ge(tensor, val), torch.max(tensor, val + eps),
                       torch.min(tensor, val - eps))
  return tensor


def safe_divide(num, den, eps=1e-10):
  """Performs a safe divide. Do not use this function.

  Args:
    num: Numerator.
    den: Denominator.
    eps: Epsilon.

  Returns:
    Quotient tensor.

  """
  new_den = clamp_away_from(den, eps=eps)
  return num / new_den


def find_torch_device(device="cuda"):
  """Detects if cuda is available.

  Args:
    device: Target device.

  Returns:
    A torch.device object representing either CPU or CUDA.

  """
  device = torch.device(device)
  if device == "cuda" and not torch.cuda.is_available():
    device = torch.device("cpu")
    print("CUDA is not available, using CPU")
  return device


def rotate_equirectangular_image(image, rot_mat, linearize_angle=np.deg2rad(10)):
  """Applies the inverse of a rotation matrix to an equirectangular image to rotate it.

  Note that the rotation is performed using bilinear interpolation so applying
  this function several times will create a blurry image.
  You should always accumulate rotations.

  Args:
    image: Input erp image as (B, H, W, C) tensor.
    rot_mat: Rotation matrices as (B, 3, 3) tensors.

  Returns:
    Rotated equirectangular image of the same resolution.

  """
  b, h, w, c = image.shape

  xx, yy = np.meshgrid((np.arange(0, w) + 0.5) * (2 * np.pi / w) + np.pi / 2,
                       (np.arange(0, h) + 0.5) * (np.pi / h))
  xyz = my_helpers.spherical_to_cartesian(xx, yy)
  xyz = torch.tensor(xyz, device=image.device, dtype=torch.float32)
  xyz = xyz.view(1, h, w, 3).repeat(b, 1, 1, 1)
  xyz = torch.matmul(rot_mat.view(b, 1, 1, 3, 3), xyz.view(b, h, w, 3, 1))
  xyz = xyz[:, :, :, :, 0]
  sp = cartesian_to_spherical(xyz, linearize_angle=linearize_angle)[..., :2]
  u = torch.fmod((sp[:, :, :, 0] - np.pi / 2) + 4 * np.pi, 2 * np.pi)
  u = w * u / (2 * np.pi)
  v = h * sp[:, :, :, 1] / np.pi
  u = 2 * ((u + 1) / (w + 2)) - 1
  v = 2 * (v / h) - 1
  uv = torch.stack((u, v), dim=3)
  image_extended = image.permute((0, 3, 1, 2))
  image_extended = torch.cat(
    (image_extended[:, :, :,
     -1:], image_extended, image_extended[:, :, :, :1]),
    dim=3)
  new_image = torch.nn.functional.grid_sample(image_extended,
                                              uv,
                                              mode="bilinear",
                                              padding_mode="border",
                                              align_corners=True)
  return new_image.permute((0, 2, 3, 1))


def total_params(model, trainable=False):
  """Calculates the total number of trainable parameters.

  Args:
    model: Model
    trainable: True for trainable parameters

  Returns:
    Number of parameters.
  """
  if trainable:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  else:
    params = sum(p.numel() for p in model.parameters())
  return params


def equirectangular_to_cubemap(panos, side, width=256, height=256, flipx=False):
  """Create a cubemap side from an equirectangular image.

  This function is non-differentiable.

  Args:
    panos: Panos as (B, H, W, C) tensor.
    side: Side as 0-5 number.

  Returns:
    Panos as (B, H, W, C) tensor.

  """
  batch_size, erp_height, erp_width, channels = panos.shape

  u = np.linspace(0, 1, width)
  if flipx:
    u = np.linspace(1, 0, width)
  v = np.linspace(0, 1, height)
  u, v = np.meshgrid(u, v)
  uv = np.stack((u, v), axis=-1)
  sph = my_helpers.cubemap_to_spherical(uv, side=side)
  u = np.fmod(sph[:, :, 0] + 2 * np.pi, 2 * np.pi) / (2 * np.pi)
  v = sph[:, :, 1] / np.pi
  grid = np.stack((2 * u - 1, 2 * v - 1), axis=-1)
  grid = torch.tensor(grid, dtype=torch.float32, device=panos.device)
  grid = grid.view((1, height, width, 2)).expand((batch_size, height, width, 2))
  cube_img = torch.nn.functional.grid_sample(panos.permute((0, 3, 1, 2)),
                                             grid,
                                             mode="bilinear",
                                             align_corners=True)
  return cube_img.permute((0, 2, 3, 1))


def equirectangular_to_cylindrical_grid(panos, cylinder_length=10.0, width=256,
                                        height=256,
                                        linearize_angle=np.deg2rad(15),
                                        rect_rots=None):
  """Create a cylindrical image from an equirectangular image.

  Args:
    panos: Panos as (B, H, W, C) tensor.
    cylinder_length: Length of the cylinder (where sphere is 1).
    width: Width of the output image.
    height: Height of the output image.
    linearize_angle: Linearize angle.
    rect_rots: Rotation for rectification.

  Returns:
    Grid as (B, H, W, 2) tensor.

  """
  batch_size, erp_height, erp_width, channels = panos.shape

  theta = (torch.arange(0, width, device=panos.device,
                        dtype=torch.float32) + 0.5) * (2 * np.pi / width)
  theta = theta + np.pi / 2
  y = torch.linspace(cylinder_length / 2.0, -cylinder_length / 2.0, height,
                     device=panos.device, dtype=torch.float32)
  y, theta = torch.meshgrid(y, theta)

  x = torch.cos(theta)
  z = torch.sin(theta)

  xyz = torch.stack((x, y, z), dim=-1)
  raw_xyz = xyz.view((1, height, width, 3)).expand(
    (batch_size, height, width, 3))
  rotated_xyz = raw_xyz
  if rect_rots is not None:
    # rect_rots_inv = torch.inverse(rect_rots)
    rotated_xyz = rect_rots[:, None, None, :, :] @ rotated_xyz[:, :, :, :, None]
    rotated_xyz = rotated_xyz[:, :, :, :, 0]

  spherical_coords = cartesian_to_spherical(rotated_xyz,
                                            linearize_angle=linearize_angle)

  u = torch.fmod(spherical_coords[..., 0] - np.pi / 2 + 4 * np.pi,
                 2 * np.pi) / (2 * np.pi)
  u = (u * width - 0.5 + 1) / (width + 2)
  u = 2.0 * u - 1.0
  v = 2.0 * spherical_coords[..., 1] / np.pi - 1.0
  grid = torch.stack((u, v), dim=-1)
  return grid, rotated_xyz, raw_xyz


def equirectangular_to_cylindrical(panos, cylinder_length=10.0, width=256,
                                   height=256,
                                   linearize_angle=np.deg2rad(15),
                                   rect_rots=None,
                                   depth=False):
  """Create a cylindrical image from an equirectangular image.

  Args:
    panos: Panos as (B, H, W, C) tensor.
    cylinder_length: Length of the cylinder (where sphere is 1).
    width: Width of the output image.
    height: Height of the output image.
    linearize_angle: Linearize angle.
    rect_rots: Rotation for rectification.
    depth: Flag for depth images.

  Returns:
    Panos as (B, H, W, C) tensor.

  """
  linearize_angle = np.deg2rad(0)
  grid, rotated_xyz, raw_xyz = equirectangular_to_cylindrical_grid(
    panos=panos,
    cylinder_length=cylinder_length,
    width=width,
    height=height,
    linearize_angle=linearize_angle,
    rect_rots=rect_rots)

  expanded_panos = panos.permute((0, 3, 1, 2))
  expanded_panos = torch.cat((expanded_panos[:, :, :, 0:1],
                              expanded_panos,
                              expanded_panos[:, :, :, -1:]), dim=3)
  cylinder_image = torch.nn.functional.grid_sample(expanded_panos,
                                                   grid,
                                                   mode="bilinear",
                                                   align_corners=False)

  if depth:
    # depths = torch.nn.functional.grid_sample(
    #   panos.permute((0, 3, 1, 2)),
    #   grid,
    #   mode="bilinear",
    #   align_corners=False)
    depths = cylinder_image.permute((0, 2, 3, 1))
    spherical_coords = cartesian_to_spherical(raw_xyz)
    cylinder_depths = depths * torch.cos(spherical_coords[..., 1] - np.pi / 2)[
                               :, :,
                               :, None]
    return cylinder_depths

  return cylinder_image.permute((0, 2, 3, 1))


def cylindrical_to_equirectangular_grid(panos, cylinder_length=10.0, width=256,
                                        height=256,
                                        linearize_angle=np.deg2rad(15),
                                        rect_rots=None):
  """Cylindrical to equirectangular grid.

  Args:
    panos: Panoramas as (B, H, W, C) tensor.
    cylinder_length: Cylinder length.
    width: Width.
    height: Height
    linearize_angle: Linearize angle.
    rect_rots: Rotations for rectification.

  Returns:
    UV coords [-1, 1]
  """
  batch_size, cyl_height, cyl_width, channels = panos.shape

  theta = (torch.arange(0, width, device=panos.device,
                        dtype=torch.float32) + 0.5) * (2 * np.pi / width)
  theta = theta + np.pi / 2
  phi = (torch.arange(0, height, device=panos.device,
                      dtype=torch.float32) + 0.5) * (np.pi / height)
  phi, theta = torch.meshgrid(phi, theta)
  raw_xyz = spherical_to_cartesian(theta, phi)
  raw_xyz = safe_divide(raw_xyz,
                        torch.norm(raw_xyz[:, :, [0, 2]], dim=2, keepdim=True))
  raw_xyz = raw_xyz.view((1, height, width, 3)).expand(
    (batch_size, height, width, 3))
  rotated_xyz = raw_xyz

  if rect_rots is not None:
    rect_rots_inv = torch.inverse(rect_rots)
    rotated_xyz = rect_rots_inv[:, None, None, :, :] @ \
                  rotated_xyz[:, :, :, :, None]
    rotated_xyz = rotated_xyz[:, :, :, :, 0]

  spherical_coords = cartesian_to_spherical(rotated_xyz,
                                            linearize_angle=linearize_angle)
  u = torch.fmod(spherical_coords[..., 0] - np.pi / 2 + 4 * np.pi, 2 * np.pi)
  u = u / np.pi - 1.0
  v = -rotated_xyz[..., 1] / (cylinder_length / 2)
  grid = torch.stack((u, v), dim=-1)
  return grid, rotated_xyz, raw_xyz, spherical_coords


def cylindrical_to_equirectangular(panos, cylinder_length=10.0, width=256,
                                   height=256,
                                   linearize_angle=np.deg2rad(15),
                                   rect_rots=None, depth=False):
  """Cylindrical image to ERP image.

  Args:
    panos: Panoramas as (B, H, W, C) tensor.
    cylinder_length: Cylinder length.
    width: Width.
    height: Height.
    linearize_angle: Linearize angle.
    rect_rots: Rotations for rectification.
    depth: Convert cylindrical depth to ERP depth.

  Returns:
    ERP image as (B, H, W, C) tensor.
  """
  linearize_angle = np.deg2rad(0)
  grid, rotated_xyz, raw_xyz, spherical_coords = cylindrical_to_equirectangular_grid(
    panos=panos,
    cylinder_length=cylinder_length,
    width=width,
    height=height,
    linearize_angle=linearize_angle,
    rect_rots=rect_rots)

  if depth:
    cyl_depths = torch.nn.functional.grid_sample(
      panos.permute((0, 3, 1, 2)),
      grid,
      mode="bilinear",
      align_corners=False)
    cyl_depths = cyl_depths.permute((0, 2, 3, 1))
    cyl_depths = safe_divide(
      cyl_depths,
      torch.cos(spherical_coords[..., 1] - np.pi / 2)[:, :, :, None])
    return cyl_depths

  erp_image = torch.nn.functional.grid_sample(panos.permute((0, 3, 1, 2)),
                                              grid,
                                              mode="bilinear",
                                              align_corners=False)
  return erp_image.permute((0, 2, 3, 1))


def calculate_cost_volume(images,
                          stride=1.0,
                          slide_range=1.0,
                          cost_type="abs_diff",
                          direction="up"):
  """Calculates a cost volume from the images.

  Args:
    images: Tensor of shape (B, 2, H, W, C).
      The target image should be in index 1 along dim 1.
    stride: Stride to move the tensor. Can be float.
    slide_range: How far to move the tensor. Can be float.
    cost_type: Type of the cost volume.
    direction: Direction of cost volume.

  Returns:
    Tensor of shape (B, L, H, W, C).
  """
  batch_size, image_ct, height, width, channels = images.shape
  other_images_cl = images[:, 0].permute((0, 3, 1, 2))
  reference_image = images[:, 1]
  cost_volume = []
  grid_x = torch.linspace(-1, 1, width, device=images.device)
  grix_y = torch.linspace(-1, 1, height, device=images.device)
  grid_y, grid_x = torch.meshgrid(grix_y, grid_x)
  for i in np.arange(0, slide_range * height, stride):
    if direction == "up":
      m_grid_x = grid_x
      m_grid_y = grid_y + i * 2 / height
    elif direction == "left":
      m_grid_x = grid_x + i * 2 / width
      m_grid_y = grid_y
    m_grid = torch.stack((m_grid_x, m_grid_y), dim=2).expand(
      (batch_size, height, width, 2))
    other_image = torch.nn.functional.grid_sample(other_images_cl,
                                                  m_grid,
                                                  mode="bilinear",
                                                  align_corners=False)
    other_image = other_image.permute((0, 2, 3, 1))
    if cost_type == "abs_diff":
      diff_image = torch.abs(reference_image - other_image)
    cost_volume.append(diff_image)
  cost_volume = torch.stack(cost_volume, dim=1)
  return cost_volume
