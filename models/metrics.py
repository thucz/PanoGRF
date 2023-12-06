import numpy as np
import torch
import torch.nn.functional as F


class WSPSNR:
  """Weighted to spherical PSNR"""

  def __init__(self):
    self.weight_cache = {}

  def get_weights(self, height=1080, width=1920):
    """Gets cached weights.

    Args:
        height: Height.
        width: Width.

    Returns:
      Weights as H, W tensor.

    """
    key = str(height) + ";" + str(width)
    if key not in self.weight_cache:
      v = (np.arange(0, height) + 0.5) * (np.pi / height)
      v = np.sin(v).reshape(height, 1)
      v = np.broadcast_to(v, (height, width))
      self.weight_cache[key] = v.copy()
    return self.weight_cache[key]

  def calculate_wsmse(self, reconstructed, reference):
    """Calculates weighted mse for a single channel.

    Args:
        reconstructed: Image as B, H, W, C tensor.
        reference: Image as B, H, W, C tensor.

    Returns:
        wsmse
    """
    batch_size, height, width, channels = reconstructed.shape
    weights = torch.tensor(self.get_weights(height, width),
                           device=reconstructed.device,
                           dtype=reconstructed.dtype)
    weights = weights.view(1, height, width, 1).expand(
      batch_size, -1, -1, channels)
    squared_error = torch.pow((reconstructed - reference), 2.0)
    wmse = torch.sum(weights * squared_error, dim=(1, 2, 3)) / torch.sum(
      weights, dim=(1, 2, 3))
    return wmse

  def ws_psnr(self, y_pred, y_true, max_val=1.0):
    """Weighted to spherical PSNR.

    Args:
      y_pred: First image as B, H, W, C tensor.
      y_true: Second image.
      max: Maximum value.

    Returns:
      Tensor.

    """
    wmse = self.calculate_wsmse(y_pred, y_true)
    ws_psnr = 10 * torch.log10(max_val * max_val / wmse)
    return ws_psnr


class SSSIM:
  """Spherical SSIM
  Methods to predict the SSIM, taken from
  https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
  """

  def __init__(self):
    self.weight_cache = {}

  def wrap_padding_2d(self, x_in, padding=(1, 1)):
    """Does wrap padding.

    Args:
      x_in: Input as B, C, H, W tensor.
      padding: Tuple of padding sizes.

    Returns:
      Padded tensor.

    """
    if isinstance(padding, int):
      padding = (padding, padding)
    batch_size, channels, height, width = x_in.shape
    x = x_in
    if padding[0]:
      zeros_l = torch.zeros((batch_size, channels, padding[0], width),
                            dtype=x_in.dtype,
                            device=x_in.device)
      x = torch.cat((zeros_l, x, zeros_l), dim=2)
    if padding[1]:
      left_padding = x[:, :, :, -padding[1]:]
      right_padding = x[:, :, :, :padding[1]]
      x = torch.cat((left_padding, x, right_padding), dim=3)
    return x

  def get_weights(self, height=1080, width=1920, dtype=torch.float32):
    """Gets cached weights.

    Args:
        height: Height.
        width: Width.
        dtype: Dtyle

    Returns:
      Weights as H, W tensor.

    """
    key = str(height) + ";" + str(width)
    if key not in self.weight_cache:
      v = (np.arange(0, height) + 0.5) * (np.pi / height)
      v = np.sin(v).reshape(height, 1)
      v = np.broadcast_to(v, (height, width))
      self.weight_cache[key] = v
    return torch.tensor(self.weight_cache[key], dtype=dtype)

  def gaussian(self, window_size, sigma):
    gauss = torch.Tensor(
      [
        np.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
        for x in range(window_size)
      ]
    )
    return gauss / gauss.sum()

  def create_window(self, window_size, channel):
    _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(
      _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window

  def _sssim(self,
             img1, img2, window, window_size, channel, mask=None,
             size_average=True
             ):
    batch_size, channels, height, width = img1.shape
    mu1 = F.conv2d(self.wrap_padding_2d(img1, padding=window_size // 2),
                   window, groups=channel)
    mu2 = F.conv2d(self.wrap_padding_2d(img2, padding=window_size // 2), window,
                   groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(self.wrap_padding_2d(img1 * img1, padding=window_size // 2),
                 window,
                 groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(self.wrap_padding_2d(img2 * img2, padding=window_size // 2),
                 window,
                 groups=channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(self.wrap_padding_2d(img1 * img2, padding=window_size // 2),
                 window,
                 groups=channel)
        - mu1_mu2
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    weights = self.get_weights(height, width).view(1, 1, height, width).expand(
      batch_size, channels, -1, -1)
    print("ssim_map.shape", ssim_map.shape)
    print("Weights shape", weights.shape)
    exit(0)

    if mask is not None:
      raise Exception("mask not none is Unimplemented")
      # b = mask.size(0)
      # ssim_map = ssim_map.mean(dim=1, keepdim=True) * mask
      # ssim_map = ssim_map.view(b, -1).sum(dim=1) / mask.view(b, -1).sum(
      #   dim=1
      # ).clamp(min=1)
      # return ssim_map

    if size_average:
      return ssim_map.mean()
    else:
      return ssim_map.mean((1, 2, 3))

  def s_ssim(self, y_pred, y_true, window_size=11, mask=None,
             size_average=True):
    """Spherical SSIM.

    Args:
      y_pred: First image as B, H, W, C tensor.
      y_true: Second image.
      max: Maximum value.

    Returns:
      Tensor.

    """
    raise Exception("S-SSIM is not fully implemented")
    # batch_size, height, width, channels = y_pred.size()
    # window = self.create_window(window_size, channels)
    #
    # if y_pred.is_cuda:
    #   window = window.cuda(y_pred.get_device())
    # window = window.type_as(y_pred)
    #
    # return self._sssim(y_pred.permute(0, 3, 1, 2), y_true.permute(0, 3, 1, 2),
    #                    window, window_size, channels, mask,
    #                    size_average)
