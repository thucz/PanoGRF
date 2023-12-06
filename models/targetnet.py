# Lint as: python3
"""This file contains the TargetNet which predicts target depth based on
the source depth and pose. This is monocular.
"""

import torch
from torch import nn
from models.common_blocks import ConvBlock
from kornia.geometry.conversions import rotation_matrix_to_quaternion
import numpy as np


class TargetNet(nn.Module):
  """A UNet for target depth prediction.
  """

  def __init__(self,
               input_channels=13,
               output_channels=1,
               layers=7,
               width=256,
               height=256,
               device="cuda",
               **kwargs):
    """Create an TargetNet.

    Args:
      input_channels: Number of input channels.
      output_channels: Number of output channels.
      layers: Number of layers.
      device: Cuda device.
      **kwargs:
    """
    super().__init__(**kwargs)

    self.layers = layers
    encoders = [
        ConvBlock(in_channels=input_channels,
                  out_channels=16,
                  kernel_size=4,
                  stride=2,
                  padding=1,
                  upscale=False,
                  use_wrap_padding=True)
    ]
    decoders = [
        ConvBlock(in_channels=32,
                  out_channels=16,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  upscale=True,
                  use_wrap_padding=True)
    ]

    for i in range(1, self.layers):
      channels = 2**(i + 3)
      encoders.append(
          ConvBlock(in_channels=channels,
                    out_channels=2 * channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    upscale=False,
                    use_wrap_padding=True))
      decoders.append(
          ConvBlock(in_channels=4 * channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    upscale=True,
                    use_wrap_padding=True))

    self.encoders = nn.ModuleList(encoders)
    self.decoders = nn.ModuleList(decoders)
    # self.final_conv = nn.Conv2d(16,
    #                             output_channels,
    #                             kernel_size=3,
    #                             stride=1,
    #                             padding=1)
    self.final_conv = ConvBlock(in_channels=16,
                                out_channels=output_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                upscale=False,
                                gate=False,
                                use_wrap_padding=True,
                                use_batch_norm=False,
                                use_activation=False)

  def disp_to_depth(self, disp, min_depth=0.1, max_depth=100):
    """Disparity to depth from monodepth2
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth

  def depth_to_disp(self, depth, min_depth=0.1, max_depth=100):
    """Depth to disparity.

    Args:
      depth: Depth.
      min_depth: Min depth.
      max_depth: Max depth.

    Returns: Disparity.

    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp = (scaled_disp - min_disp) / (max_disp - min_disp)
    return disp

  def forward(self, input, **kwargs):
    """Forward pass through the TargetNet network.

    Args:
      input: Input depth, rotation, and translation
        Depth should be BxHxWxC.
        Rotation should be Bx3x3.
        Translation should be Bx3.
      **kwargs:

    Returns: Output of the forward pass.

    """

    # Change to channels first.
    EPS = 0.00001
    depth_pano, rot, trans = input

    batch_size, height, width, channels = depth_pano.shape
    input_depth = depth_pano.permute((0, 3, 1, 2))
    trans = trans[:, :, None, None].expand((batch_size, 3, height, width))
    rot = rot.reshape((batch_size, 9, 1, 1)).expand(
        (batch_size, 9, height, width))

    input_depth = self.depth_to_disp(input_depth, 0.1, 100)
    # Concatenate depth, rotation, and translation along the channel axis.
    x = torch.cat((input_depth, rot, trans), dim=1)
    x_all = []
    # Encode.
    for i in range(self.layers):
      x = self.encoders[i](x)
      x_all.append(x)

    # Decode with skip connections.
    for i in range(self.layers - 1, -1, -1):
      x = torch.cat((x, x_all[i]), dim=1)
      x = self.decoders[i](x)
    x = self.final_conv(x)
    x = torch.sigmoid(x)

    # Change back to channels last.
    x = x.permute((0, 2, 3, 1))
    return self.disp_to_depth(x, 0.1, 100)

  def compute_l1_loss(self, y_pred, y_true):
    """Compute the l1 loss between the rendered image and the ground truth.

    Args:
      y_pred: Predicted image. Channels last format. BxHxWxC.
      y_true: Ground truth image. Channels last format.

    Returns: Loss.

    """
    height, width = y_pred.shape[1:3]
    # sin_phi = torch.tensor(np.sin(np.arange(height) * np.pi / height),
    #                        dtype=y_pred.dtype,
    #                        device=y_pred.device)
    # sin_phi = sin_phi.view((
    #     height,
    #     1,
    # )).repeat((1, width)).view((1, height, width, 1))
    loss = torch.abs(y_true - y_pred)
    loss = torch.mean(loss)
    return loss
