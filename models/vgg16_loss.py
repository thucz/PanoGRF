# Lint as: python3
"""This file contains a layer to extract VGG features.
It is useful for computing a VGG perceptual loss.
"""

import torch
import torchvision
from torch import nn
from models.common_blocks import ConvBlock
from collections import namedtuple

LossOutput = namedtuple("LossOutput",
                        ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])


class LossNetwork(torch.nn.Module):
  """Class to help calculate VGG perceptual losses.

  """

  def __init__(self):
    """Create a Loss Network.

    """
    super().__init__()

    self.vgg16 = torchvision.models.vgg16(pretrained=True)
    self.vgg_layers = self.vgg16.features
    self.layer_name_mapping = {
        '3': "relu1_2",
        '8': "relu2_2",
        '15': "relu3_3",
        '22': "relu4_3"
    }

  def forward(self, x, layer=None):
    """Returns VGG 16 features.

    Args:
      x: Input image

    Returns:
      Feature map

    """
    output = {}
    for name, module in self.vgg_layers._modules.items():
      x = module(x)
      if name in self.layer_name_mapping:
        output[self.layer_name_mapping[name]] = x
      if layer is not None and layer == self.layer_name_mapping[name]:
        return x
    return LossOutput(**output)

  def calculate_l1_vgg_loss(self, y_pred, y_true, layer="relu2_2"):
    """Calculate perceptual loss of the input image.

    Args:
      y_pred: Predicted image.
      y_true: True image.
      layer: Which feature layer of the vgg to use.

    Returns: Loss as a tensor.

    """
    y_pred = y_pred.permute((0, 3, 1, 2))
    y_true = y_true.permute((0, 3, 1, 2))
    y_pred_features = self.forward(y_pred, layer)
    y_true_features = self.forward(y_true, layer)
    print("y_pred_features", y_pred_features.shape)
    return torch.mean(torch.abs(y_pred_features - y_true_features))

  def calculate_l2_vgg_loss(self, y_pred, y_true, layer="relu2_2"):
    """Calculate perceptual loss of the input image.

    Args:
      y_pred: Predicted image.
      y_true: True image.
      layer: Which feature layer of the vgg to use.

    Returns: Loss as a tensor.

    """
    y_pred = y_pred.permute((0, 3, 1, 2))
    y_true = y_true.permute((0, 3, 1, 2))
    y_pred_features = self.forward(y_pred, layer)
    y_true_features = self.forward(y_true, layer)
    return torch.mean(torch.pow(y_pred_features - y_true_features, 2))
