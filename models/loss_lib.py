# Lint as: python3
"""This module contains a various loss functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.architecture import VGG19
from helpers import my_torch_helpers


def compute_l1_loss(y_pred, y_true, normalize=False, keep_batch=False):
  """Computes the l1 loss between the rendered image and the ground truth.

  Args:
    y_pred: Predicted image as a (B, H, W, C) tensor.
    y_true: Ground truth image as a (B, H, W, C) tensor.
    normalize: Whether to normalize the scale of the inputs.

  Returns:
     Loss tensor.

  """
  sum_axis = (0, 1, 2, 3)
  if keep_batch:
    sum_axis = (1, 2, 3)
  if normalize:
    y_pred = normalize_depth(y_pred)
    y_true = normalize_depth(y_true)
  loss = torch.abs(y_true - y_pred)
  loss = torch.mean(loss, dim=sum_axis)
  return loss


def compute_l1_sphere_loss(y_pred, y_true, mask=None, keep_batch=False):
  """Computes the l1 loss between the rendered image and the ground truth
  with a sin factor to account for the size of each pixel.

  Args:
    y_pred: Predicted image as a (B, H, W, C) tensor.
    y_true: Ground truth image as a (B, H, W, C) tensor.
    mask: Mask for valid GT values.

  Returns:
     Loss tensor.

  """
  batch_size, height, width, channels = y_pred.shape
  sin_phi = torch.arange(0, height, dtype=y_pred.dtype, device=y_pred.device)
  sin_phi = torch.sin((sin_phi + 0.5) * np.pi / height)
  sum_axis = (0, 1, 2, 3)
  if keep_batch:
    sum_axis = (1, 2, 3)
  if mask is not None:
    sin_phi = sin_phi.view(1, height, 1, 1).expand(batch_size, height, width,
                                                   channels)
    sin_phi = sin_phi * mask
    loss = torch.abs(y_true - y_pred) * sin_phi
    loss = torch.sum(loss, dim=sum_axis) / torch.sum(sin_phi, dim=sum_axis)
  else:
    sin_phi = sin_phi.view(1, height, 1, 1).expand(batch_size, height, width, 1)
    loss = torch.abs(y_true - y_pred) * sin_phi
    loss = torch.sum(loss, dim=sum_axis) / (
        channels * torch.sum(sin_phi, dim=sum_axis))
  return loss


def compute_l2_sphere_loss(y_pred, y_true, keep_batch=False):
  """Computes the squared l2 loss between the rendered image and the ground truth
  with a sin factor to account for the size of each pixel.

  Args:
    y_pred: Predicted image as a (B, H, W, C) tensor.
    y_true: Ground truth image as a (B, H, W, C) tensor.

  Returns:
     Loss tensor.

  """
  sum_axis = (0, 1, 2, 3)
  if keep_batch:
    sum_axis = (1, 2, 3)
  batch_size, height, width, channels = y_pred.shape
  sin_phi = torch.arange(0, height, dtype=y_pred.dtype, device=y_pred.device)
  sin_phi = torch.sin((sin_phi + 0.5) * np.pi / height)
  sin_phi = sin_phi.view(1, height, 1, 1).expand(batch_size, height, width, 1)
  loss = torch.pow(y_true - y_pred, 2.0) * sin_phi
  loss = torch.sum(loss, dim=sum_axis) / (
        channels * torch.sum(sin_phi, dim=sum_axis))
  return loss

# Loss for training D-Net
# depths_pred, gt_dmap, sigma_pred, torch.gt(gt_dmap.permute((0, 2, 3, 1)), 0.1))#0.01 * torch.nn.functional.gaussian_nll_loss()
def new_compute_gaussian_loss(depths_pred, gt_dmap, sigma_pred, gt_depth_mask, sphere=False, keep_batch=False):
  # depth_loss = 0.01 * torch.nn.functional.gaussian_nll_loss(pred[0][valid_target], target[valid_target], pred[1][valid_target].pow(2))
  if sphere:
      sum_axis = (0, 1, 2, 3)
      if keep_batch:
        sum_axis = (1, 2, 3)
      
      batch_size, channels, height, width = pred.shape
      sin_phi = torch.arange(0, height, dtype=pred.dtype, device=pred.device)
      sin_phi = torch.sin((sin_phi + 0.5) * np.pi / height)
      # sum_axis = (0, 1, 2, 3)
      # if keep_batch:
      #   sum_axis = (1, 2, 3)

      if gt_depth_mask is not None:
        sin_phi = sin_phi.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        sin_phi = sin_phi * gt_depth_mask
      nll = torch.nn.functional.gaussian_nll_loss(depths_pred, gt_dmap, sigma_pred.pow(2), reduction="none")*sin_phi
      loss = torch.sum(nll, dim=sum_axis) / (
            channels * torch.sum(sin_phi, dim=sum_axis))
  else:
    # import ipdb;ipdb.set_trace()
    depths_pred = depths_pred[gt_depth_mask]
    gt_dmap = gt_dmap[gt_depth_mask]
    sigma_pred = sigma_pred[gt_depth_mask]
    # import ipdb;ipdb.set_trace()
    loss = 0.01 * torch.nn.functional.gaussian_nll_loss(depths_pred, gt_dmap, sigma_pred.pow(2))

    # loss = nll.mean()
    # los
  return loss
      

class BerhuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerhuLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target, pred, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask *= mask.detach()

        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        delta = self.threshold * torch.max(diff).data.cpu().numpy()

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0*delta ** 2, 0.)
        part2 = part2 / (2. * delta)
        diff = part1 + part2
        loss = diff.mean()
        return loss
def new_loss_uncertainty(pred, gt_depth, gt_depth_mask):
  mu, var = torch.split(pred, 1, dim=1)  # (B, 1, H, W)
  gt_depth = gt_depth[gt_depth_mask]          
  mu = mu[gt_depth_mask]          
  var = var[gt_depth_mask]
  # var[var < 1e-10] = 1e-10
  # nll = (torch.square(mu - gt_depth) / (2 * var)) + (0.5 * torch.log(var))
  return 0.01 * torch.nn.functional.gaussian_nll_loss(mu, gt_depth, var)
  # return torch.mean(nll)

def loss_uncertainty(pred, gt_depth, gt_depth_mask, sphere=False, keep_batch=False):
    if sphere:
      sum_axis = (0, 1, 2, 3)
      if keep_batch:
        sum_axis = (1, 2, 3)
      
      batch_size, channels, height, width = pred.shape
      sin_phi = torch.arange(0, height, dtype=pred.dtype, device=pred.device)
      sin_phi = torch.sin((sin_phi + 0.5) * np.pi / height)
      # sum_axis = (0, 1, 2, 3)
      # if keep_batch:
      #   sum_axis = (1, 2, 3)

      if gt_depth_mask is not None:
        sin_phi = sin_phi.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        sin_phi = sin_phi * gt_depth_mask

      # mu, var = torch.split(pred, 1, dim=1)  # (B, 1, H, W)
      

      mu = pred[:, :1, ...]
      var = pred[:, 1:, ...]
      # import ipdb;ipdb.set_trace()

      # gt_depth = gt_depth[gt_depth_mask]          
      # mu = mu[gt_depth_mask]          
      # var = var[gt_depth_mask]
      # sin_phi = sin_phi[gt_depth_mask]
      
      var[var < 1e-10] = 1e-10
      nll = ((torch.square(mu - gt_depth) / (2 * var)) + (0.5 * torch.log(var)))*sin_phi
      # loss = torch.sum(nll)/torch.sum(sin_phi)
      loss = torch.sum(nll, dim=sum_axis) / (
            channels * torch.sum(sin_phi, dim=sum_axis))
      return loss
    else:
      mu, var = torch.split(pred, 1, dim=1)  # (B, 1, H, W)
      gt_depth = gt_depth[gt_depth_mask]          
      mu = mu[gt_depth_mask]          
      var = var[gt_depth_mask]
      var[var < 1e-10] = 1e-10
      nll = (torch.square(mu - gt_depth) / (2 * var)) + (0.5 * torch.log(var))
      return torch.mean(nll)
    # else:
    #     raise Exception

def normalize_depth(depth, new_std=1, new_mean=0):
  """Normalizes the mean and std dev of each input.

  Args:
    depth: Depth image as (B, H, W, 1) tensor.
    new_std: New standard deviation as (B, 1, 1, 1) tensor.
    new_mean: New mean as (B, 1, 1, 1) tensor

  Returns:
    Normalized depth as a (B, H, W, 1) tensor.

  """
  std, mean = torch.std_mean(depth, dim=(1, 2), keepdim=True)
  depth = (depth - mean) / std
  depth = new_std * depth + new_mean
  return depth


def compute_l2_loss(y_pred, y_true, keep_batch=False):
  """Computes the l2 loss.

  Args:
    y_pred: Predicted image.
    y_true: True image.

  Returns:
    The loss.

  """
  sum_axis = (0, 1, 2, 3)
  if keep_batch:
    sum_axis = (1, 2, 3)
  loss = torch.pow(y_pred - y_true, 2.0)
  loss = torch.mean(loss, dim=sum_axis)
  return loss


def compute_masked_l1_loss(y_pred, y_true):
  """Computes the l1 loss between the rendered image and the ground truth.

  Args:
    y_pred: Predicted image.
    y_true: Ground truth image.

  Returns:
    Loss.

  """
  height, width = y_pred.shape[1:3]
  sin_phi = torch.tensor(np.sin(np.arange(height) * np.pi / height),
                         dtype=y_pred.dtype,
                         device=y_pred.device)
  sin_phi = sin_phi.view((height, 1)).repeat((1, width)).view(
    (1, height, width, 1))
  valid_pixels = torch.gt(torch.sum(y_pred, -1), 0.01).type(torch.float32)
  loss = torch.abs(y_true - y_pred) * valid_pixels[:, :, :, None]
  # loss = loss * sin_phi
  loss = torch.sum(loss, dim=(1, 2, 3))
  loss = loss / torch.sum(valid_pixels, dim=(1, 2))
  loss = torch.mean(loss)
  return loss


def compute_downsampled_loss(y_pred, y_true, size):
  """Downsamples using SAT and then calculates l1 loss.

  Args:
    y_pred: Predicted image
    y_true: True image
    size: Downsampled size

  Returns:
    Loss

  """
  downsampled_y_pred = my_torch_helpers.sat_downsample(y_pred, size=size)
  downsampled_y_true = my_torch_helpers.sat_downsample(y_true, size=size)
  return torch.mean(torch.abs(downsampled_y_pred - downsampled_y_true))


def compute_monodepth2_loss(y_pred_org, y_pred, y_true):
  """Computes an automasked monodepth 2 loss.

  Args:
    y_pred_org: Original unwarped image.
    y_pred: Predicted warped image.
    y_true: Ground truth image.

  Returns:
    loss as a single value tensor.

  """
  # height, width = y_pred.shape[1:3]
  loss_pred = torch.mean(torch.abs(y_pred - y_true), dim=3)
  loss_orig = torch.mean(torch.abs(y_pred_org - y_true), dim=3)
  loss = torch.min(loss_pred, loss_orig)
  loss = torch.mean(loss)
  return loss


def compute_mixed_monodepth2_l1_loss(y_pred_org, y_pred, y_true):
  """Mixes monodepth2 loss and a basic l1 loss. The top half of the image uses an l1 loss and the bottom half
  uses the auto masked loss from monodepth2.

  Args:
    y_pred_org: Original unwarped image as BxHxWx3 tensor.
    y_pred: Predicted image as BxHxWx3 tensor.
    y_true: True image as BxHxWx3 tensor.

  Returns:
    loss as a single value tensor.

  """
  height, width = y_pred.shape[1:3]
  height_2 = int(height / 2)
  top_loss = compute_l1_loss(y_pred[:, :height_2, :, :],
                             y_true[:, :height_2, :, :])
  bottom_loss = compute_monodepth2_loss(y_pred_org[:, height_2:, :, :],
                                        y_pred[:, height_2:, :, :],
                                        y_true[:, height_2:, :, :])
  loss = top_loss + bottom_loss
  return loss


def compute_masked_downsampled_loss(y_pred, y_true, size):
  """Downsamples using SAT and then calculates l1 loss.
  Masks out black pixels.

  Args:
    y_pred: Predicted image
    y_true: True image
    size: Downsampled size

  Returns:
    Loss

  """
  valid_pixels = torch.gt(torch.sum(y_pred, -1),
                          0.01).type(torch.float32)[:, :, :, None]
  downsampled_y_pred = my_torch_helpers.masked_sat_downsample(
    y_pred, size=size, valid_pixels=valid_pixels)
  downsampled_y_true = my_torch_helpers.sat_downsample(y_true, size=size)
  return torch.mean(torch.abs(downsampled_y_pred - downsampled_y_true))


def compute_m3d_depth_loss(y_pred, y_true):
  """Computes l1 loss over m3d depth images. Ignores areas with 0 depth."""
  valid_pixels = torch.gt(y_pred, 0.01).type(torch.float32)
  loss = torch.sum(torch.abs(y_true - y_pred) * valid_pixels,
                   dim=(1, 2)) / torch.sum(valid_pixels, dim=(1, 2))
  loss = torch.mean(loss)
  return loss


def compute_depth_smoothness_loss(depths):
  """Computes a loss based on the depth's smoothness.

  Args:
    depths: Depth image in channel last format.

  Returns:
    Tensor containing the loss.

  """
  gradient_y = torch.abs(depths[:, :-1, :, :] - depths[:, 1:, :, :])
  gradient_x = torch.abs(depths[:, :, :-1, :] - depths[:, :, 1:, :])
  # mean_depth = torch.mean(depths, dim=(1, 2), keepdim=True)
  # mean_depth = torch.where(
  #     torch.eq(mean_depth, 0),
  #     torch.tensor(1.0, device=mean_depth.device, dtype=mean_depth.dtype),
  #     mean_depth)
  mean_gradient = torch.mean(gradient_x) + torch.mean(gradient_y)
  return mean_gradient


def compute_min_patch_loss(y_pred,
                           y_true,
                           patch_size=5,
                           stride=1,
                           stride_dist=3):
  """Computes the pixelwise l1 loss for different offsets of the image.
     Then splits the image into patches and computes the minimum loss for each patch.
     Returns the mean of losses across all patches.

  Args:
    y_pred: Predicted images as BxHxWxC format
    y_true: True images as BxHxWxC format.
    patch_size: Approximate size of patches.
    stride: Distance of patch shifts.
    stride_dist: Maximum offset in each direction.

  Returns:
    Tensor containing the loss.

  """
  batch_size, height, width, channels = y_pred.shape
  batch_size_arr = np.arange(batch_size)
  channels_arr = np.arange(channels)
  patch_losses = []
  for offset_x in range(-stride_dist, stride_dist + 1, stride):
    for offset_y in range(-stride_dist, stride_dist + 1, stride):
      y_pred_y = np.arange(0, height)
      y_pred_x = np.arange(0, width)
      y_true_y = np.arange(0, height) + offset_y
      y_true_y = np.clip(y_true_y, 0, height - 1)
      y_true_x = np.arange(0, width) + offset_x
      y_true_x = np.clip(y_true_x, 0, width - 1)
      y_pred_subarr = y_pred[np.ix_(batch_size_arr, y_pred_y, y_pred_x,
                                    channels_arr)]
      y_true_subarr = y_true[np.ix_(batch_size_arr, y_true_y, y_true_x,
                                    channels_arr)]
      l1_loss = torch.mean(torch.abs(y_pred_subarr - y_true_subarr), dim=3)
      downsampled_l1_loss = my_torch_helpers.sat_downsample(
        l1_loss[:, :, :, None],
        size=(int(width / patch_size), int(height // patch_size)))
      patch_losses.append(downsampled_l1_loss)
  patch_losses = torch.stack(patch_losses, dim=0)
  patch_losses = torch.min(patch_losses, dim=0)[0]
  return torch.mean(patch_losses)


def depth_range_penalty(depth, min=1, max=100):
  """Penalize depth outside of this range.

  Args:
    depth: Depth as a (B, H, W, 1) tensor.
    min: Minimum depth.
    max: Maximum depth.

  Returns: Loss tensor.

  """
  too_small = torch.abs(torch.clamp(depth - min, max=0))
  too_big = torch.abs(torch.clamp(depth - max, min=0))
  return torch.mean(too_small) + torch.mean(too_big)


def l1_depth_ignore_sky(y_pred, y_true, threshold=65):
  """L1 loss with 0 if predicted and GT are both > threshold

  Args:
    y_pred: Predicted depth a (B, H, W, 1) tensor
    y_true: GT depth as (B, H, W, 1) tensor.
    threshold: threshold in meters

  Returns:
    Loss tensor.
  """
  sky_region = (torch.gt(y_pred, threshold) & torch.gt(y_true, threshold)).type(
    torch.float32)
  l1_loss = torch.abs(y_pred - y_true) * (1.0 - sky_region)
  return torch.mean(l1_loss)


class VGGLoss(nn.Module):
  """https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py"""
  def __init__(self):
    super(VGGLoss, self).__init__()
    self.vgg = VGG19().cuda()
    self.criterion = nn.L1Loss()
    self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

  def forward(self, x, y):
    x_vgg, y_vgg = self.vgg(x), self.vgg(y)
    loss = 0
    for i in range(len(x_vgg)):
      loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
    return loss