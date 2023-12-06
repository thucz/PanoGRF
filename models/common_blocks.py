# Lint as: python3
"""This module contains a list of common neural network components including
conv-bn-lrelu blocks and wrap-around padding.
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
  """A basic ConvBlock consisting of conv-batchnorm-lrelu.
    Add gate=true to use add a gate to the convolution.

  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=4,
               padding=1,
               stride=2,
               upscale=False,
               gate=False,
               use_wrap_padding=False,
               use_batch_norm=True,
               use_activation=True,
               **kwargs):
    """Initializes a conv2d block.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      kernel_size: Kernel size.
      padding: Padding as a number or tuple.
      stride: Stride as a number or tuple.
      upscale: Boolean indicating whether to upscale the input.
      gate: Boolean indicating whether to use gated convolutions.
      use_wrap_padding: Boolean indicating whether to use wrap padding.
      use_batch_norm: Boolean indicating whether to include batch norm.
      use_activation: Boolean indicating whether to include lrelu activation.
      **kwargs:
    """
    super().__init__(**kwargs)

    self.use_wrap_padding = use_wrap_padding
    self.use_batch_norm = use_batch_norm
    self.use_activation = use_activation
    if use_wrap_padding:
      self.padding = WrapPadding(padding=padding)
    self.conv = nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=kernel_size,
                          padding=0 if use_wrap_padding else padding,
                          stride=stride,
                          padding_mode="zeros")
    if self.use_batch_norm:
      self.norm = nn.BatchNorm2d(num_features=out_channels)
    else:
      self.norm = nn.Identity()
    if self.use_activation:
      self.act = nn.LeakyReLU()
    else:
      self.act = nn.Identity()
    self.upscale = upscale

    self.gate = None
    if gate:
      self.gate = nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            padding_mode="zeros")

  def forward(self, input, **kwargs):
    """Forward pass."""
    x = input
    if self.upscale:
      x = torch.nn.functional.interpolate(x,
                                          scale_factor=2,
                                          mode="bilinear",
                                          align_corners=False)
    if self.use_wrap_padding:
      x = self.padding(x)
    if self.gate is not None:
      y = self.gate(x)
      y = torch.sigmoid(y)
    x = self.conv(x)
    x = self.norm(x)
    x = self.act(x)
    if self.gate is not None:
      x = x * y
    return x


class ConvBlock2(nn.Module):
  """A basic Conv2D Block consisting of conv-lrelu-conv-lrelu-pool.
    If upsample is true then this will be upsample-conv-lrelu-conv-lrelu.

  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               padding=1,
               stride=1,
               upscale=False,
               use_wrap_padding=False,
               use_activation=True,
               use_residual=False,
               pooling=nn.AvgPool2d(2),
               use_v_input=False,
               **kwargs):
    """Initializes a conv2d block.

    Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      kernel_size: Kernel size.
      padding: Padding as a number or tuple.
      stride: Stride as a number or tuple.
      upscale: Boolean indicating whether to upscale the input.
      gate: Boolean indicating whether to use gated convolutions.
      use_wrap_padding: Boolean indicating whether to use wrap padding.
      use_activation: Boolean indicating whether to include lrelu activation.
      use_residual: Boolean indicating whether to use residual connections.
      pooling: Pooling layer. Defaults to nn.AvgPool2d(2).
      **kwargs:

    Returns:
      A tuple containing the pooled value and unpooled value.
    """
    super().__init__(**kwargs)

    self.use_wrap_padding = use_wrap_padding
    self.use_activation = use_activation
    if use_wrap_padding:
      self.padding = WrapPadding(padding=padding)
    else:
      self.padding = nn.Identity()
    self.conv1 = nn.Conv2d(in_channels + (1 if use_v_input else 0),
                           out_channels,
                           kernel_size=kernel_size,
                           padding=0 if use_wrap_padding else padding,
                           stride=stride,
                           padding_mode="zeros")
    self.conv2 = nn.Conv2d(out_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=0 if use_wrap_padding else padding,
                           stride=stride,
                           padding_mode="zeros")
    if self.use_activation:
      self.act = nn.LeakyReLU()
    else:
      self.act = nn.Identity()
    if not pooling:
      pooling = nn.Identity()
    self.pooling = pooling
    self.use_residual = use_residual
    self.use_v_input = use_v_input
    self.upscale = Upscale() if upscale else nn.Identity()

  def forward(self, x_in, **kwargs):
    """Forward pass."""
    x = x_in
    if self.use_v_input:
      batch_size, channels, height, width = x.shape
      v = torch.linspace(0, 1, height, dtype=x.dtype, device=x.device) \
        .view(1, 1, height, 1) \
        .expand(batch_size, 1, height, width)
      x = torch.cat((x, v), dim=1)
    x = self.upscale(x)
    x2 = self.padding(x)
    x2 = self.conv1(x2)
    x2 = self.act(x2)
    x2 = self.padding(x2)
    x2 = self.conv2(x2)
    x2 = self.act(x2)
    if self.use_residual:
      x2 = x + x2
    pooled_x = self.pooling(x2)
    return pooled_x, x2


class UNet2(nn.Module):
  """UNet based on ConvBlock2. Performs upscaling.
  To build a Unet, create a list of encoders with layers+1 ConvBlock2.
  The last conv block should have no pooling.
  Then create a list of decoders with layers ConvBlock2.
  These should have no upsample or pooling.
  Upsampling will be handled by this UNet2.

  """

  def __init__(self, encoders, decoders, interpolation="bilinear", name=""):
    """Initializes a UNet with the given encoders and decoders.

    Args:
      encoders: nn.ModuleList of encoders.
      decoders: nn.ModuleList of decoders.
    """
    super().__init__()
    self.encoders = encoders
    self.decoders = decoders
    self.interpolation = interpolation
    self.name = name

  def forward(self, x_input):
    """Forward pass."""
    x = x_input
    x_all = []

    for i in range(len(self.encoders)):
      # print("unet x features shape", x.shape)
      x, x_unpooled = self.encoders[i](x)
      x_all.append(x_unpooled)

      # print("unet x features shape", x.shape)
    x = torch.nn.functional.interpolate(x,
                                        scale_factor=2,
                                        mode=self.interpolation,
                                        align_corners=False)
    # print("unet x features shape", x.shape)
    x, _ = self.decoders[-1](x)

    for i in range(len(self.decoders) - 2, -1, -1):
      # print("unet x features shape", x.shape)
      if self.decoders[i] is not None:
        x = torch.nn.functional.interpolate(x,
                                            scale_factor=2,
                                            mode=self.interpolation,
                                            align_corners=False)
        # print("x shape", x.shape, x_all[i].shape)
        x = torch.cat((x, x_all[i]), dim=1)
        # print("unet x features shape after cat", x.shape)
        x, _ = self.decoders[i](x)
        # print("unet x features shape after output", x.shape)

    # print("final shape", x.shape)
    return x


class Upscale(nn.Module):
  """Upscale Layer"""

  def __init__(self):
    super().__init__()

  def forward(self, input):
    return torch.nn.functional.interpolate(input,
                                           scale_factor=2,
                                           mode="bilinear",
                                           align_corners=False)


class WrapPadding(nn.Module):
  """Wraps the left and right edges of the equirectangular image.
     Top and bottom are padded with zeros.
  """

  def __init__(self, padding=1):
    """Init a wrap padding layer.

    Args:
      padding: int or tuple. If tuple, first is padding for y, second is padding for x.
    """
    super().__init__()
    self.padding = padding
    if type(padding) is int:
      self.padding = (padding, padding)

  def forward(self, x_in):
    """Applies padding. Assumes channel first format.

    Args:
      x_in: Input images of size BxCxHxW.

    Returns: Padded input images of shape BxCx(H+2P)x(W+2P).

    """
    batch, channels, height, width = x_in.shape

    zeros = torch.zeros((batch, channels, self.padding[0], width),
                        dtype=x_in.dtype,
                        device=x_in.device)
    x = torch.cat((zeros, x_in, zeros), dim=2)
    x = torch.cat(
      (x[:, :, :, -self.padding[1]:], x, x[:, :, :, :self.padding[1]]), dim=3)
    
    return x


class Conv3DBlock(nn.Module):
  """A basic ConvBlock consisting of conv-batchnorm-lrelu.
    Add gate=true to use add a gate to the convolution.

  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               padding=1,
               stride=1,
               use_wrap_padding=False,
               use_batch_norm=True,
               use_activation=True,
               use_v_input=False,
               **kwargs):
    """

    Args:
      in_channels: Input channels.
      out_channels: Output channels.
      kernel_size: Kernel size. Should be a tuple or int.
      padding: Padding.
      stride: Stride
      use_wrap_padding: Boolean for wrap padding.
      use_batch_norm: Boolean for batch norm.
      use_activation: Boolean for lrelu activation.
      **kwargs:
    """
    super().__init__(**kwargs)

    self.use_wrap_padding = use_wrap_padding
    self.use_batch_norm = use_batch_norm
    self.use_activation = use_activation
    if use_wrap_padding:
      self.padding = WrapPadding3D(padding=padding)
    else:
      self.padding = nn.Identity()
    self.conv = nn.Conv3d(in_channels + (1 if use_v_input else 0),
                          out_channels,
                          kernel_size=kernel_size,
                          padding=0 if use_wrap_padding else padding,
                          stride=stride,
                          padding_mode="zeros")
    if self.use_batch_norm:
      self.norm = nn.BatchNorm3d(num_features=out_channels)
    else:
      self.norm = nn.Identity()
    if self.use_activation:
      self.act = nn.LeakyReLU()
    else:
      self.act = nn.Identity()
    self.use_v_input = use_v_input

  def forward(self, x_in):
    """Forward pass."""
    x = x_in
    if self.use_v_input:
      batch_size, channels, depth, height, width = x.shape
      v = torch.linspace(0, 1, height, dtype=x.dtype, device=x.device) \
        .view(1, 1, 1, height, 1) \
        .expand(batch_size, 1, depth, height, width)
      x = torch.cat((x, v), dim=1)
    x = self.padding(x)
    x = self.conv(x)
    x = self.norm(x)
    x = self.act(x)
    return x


class Conv3DBlockv2(nn.Module):
  """A basic ConvBlock consisting of conv-batchnorm-lrelu.
    Add gate=true to use add a gate to the convolution.

  """

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               padding=1,
               stride=1,
               use_wrap_padding=False,
               use_batch_norm=False,
               use_activation=True,
               use_v_input=False,
               use_residual=False,
               pooling=nn.AvgPool3d(2),
               name="",
               **kwargs):
    """

    Args:
      in_channels: Input channels.
      out_channels: Output channels.
      kernel_size: Kernel size. Should be a tuple or int.
      padding: Padding.
      stride: Stride
      use_wrap_padding: Boolean for wrap padding.
      use_batch_norm: Boolean for batch norm.
      use_activation: Boolean for lrelu activation.
      **kwargs:
    """
    super().__init__(**kwargs)

    self.use_wrap_padding = use_wrap_padding
    self.use_batch_norm = use_batch_norm
    self.use_activation = use_activation
    if use_wrap_padding:
      self.padding = WrapPadding3D(padding=padding)
    else:
      self.padding = nn.Identity()
    self.conv1 = nn.Conv3d(in_channels + (1 if use_v_input else 0),
                           out_channels,
                           kernel_size=kernel_size,
                           padding=0 if use_wrap_padding else padding,
                           stride=stride,
                           padding_mode="zeros")
    self.conv2 = nn.Conv3d(out_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=0 if use_wrap_padding else padding,
                           stride=stride,
                           padding_mode="zeros")
    if self.use_batch_norm:
      self.norm = nn.BatchNorm3d(num_features=out_channels)
    else:
      self.norm = nn.Identity()
    if self.use_activation:
      self.act = nn.LeakyReLU()
    else:
      self.act = nn.Identity()
    if pooling is False:
      pooling = nn.Identity()
    self.pooling = pooling
    self.use_v_input = use_v_input
    self.use_residual = use_residual
    self.name = name

  def forward(self, x_in):
    """Forward pass."""
    x = x_in
    if self.use_v_input:
      batch_size, channels, depth, height, width = x.shape
      v = torch.linspace(0, 1, height, dtype=x.dtype, device=x.device) \
        .view(1, 1, 1, height, 1) \
        .expand(batch_size, 1, depth, height, width)
      x = torch.cat((x, v), dim=1)
    x2 = self.padding(x)
    x2 = self.conv1(x2)
    x2 = self.norm(x2)
    x2 = self.act(x2)
    x2 = self.padding(x2)
    x2 = self.conv2(x2)
    x2 = self.norm(x2)
    x2 = self.act(x2)
    pooled_x = self.pooling(x2)
    if self.use_residual:
      x2 = x + x2
    return pooled_x, x2


class WrapPadding3D(nn.Module):
  """Wraps the left and right sides of a cost volume.
      Other sides are padded with zeros.
  """

  def __init__(self, padding=(0, 1, 1)):
    """Initializes a wrap padding layer.

    Args:
      padding: Tuple of 3 values corresponding to the padding on each side.
    """
    super().__init__()
    self.padding = padding
    if padding is int:
      self.padding = (padding, padding, padding)

  def forward(self, input, **kwargs):
    """Applies padding. Assumes channel first format.

    Args:
      input: Input images of shape (N, C, D, H, W)
      **kwargs:

    Returns:
      Padded input images of shape (N, C, D+2P, H+2P, W+2P).

    """
    batch, channels, length, height, width = input.shape

    x = input
    if self.padding[0]:
      zeros_l = torch.zeros((batch, channels, self.padding[0], height, width),
                            dtype=input.dtype,
                            device=input.device)
      x = torch.cat((zeros_l, x, zeros_l), dim=2)
    if self.padding[1]:
      zeros_h = torch.zeros(
        (batch, channels, x.shape[2], self.padding[1], width),
        dtype=input.dtype,
        device=input.device)
      x = torch.cat((zeros_h, x, zeros_h), dim=3)
    if self.padding[2]:
      left_padding = x[:, :, :, :, -self.padding[2]:]
      right_padding = x[:, :, :, :, :self.padding[2]]
      x = torch.cat((left_padding, x, right_padding), dim=4)
    return x
