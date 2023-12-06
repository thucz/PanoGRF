import torch.nn as nn
import os
import torch
# from network.ops import ResidualBlock, conv3x3, conv1x1
from models.common_blocks import Upscale
from helpers import my_torch_helpers
import numpy as np
from network.omni_mvsnet.pipeline3_model import load_checkpoint
import torch.nn.functional as F

import sys
sys.path.append("./UniFuse-Unidirectional-Fusion/UniFuse")
from datasets.util import Equirec2Cube
from networks import UniFuse, Equi
from networks.convert_module import erp_convert
from networks.layers import Conv3x3, upsample, ConvBlock
from collections import OrderedDict

# def load_mvs_model(model, checkpoints_path):
#     """
#     Load model from disk
#     """
#     # path = os.path.join(load_weights_dir, "{}.pth".format("model"))
#     model_dict = model.state_dict()
#     pretrained_dict = torch.load(checkpoints_path)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     # import ipdb;ipdb.set_trace()
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#     return model


    # path = os.path.join(load_weights_dir, "{}.pth".format("model"))
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(path)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)


def extract_depth_for_init_impl(args,depth):
    rfn, _, h, w = depth.shape
    near = args["min_depth"]#depth_range[:, 0][:, None, None, None]  # rfn,1,1,1
    far = args["max_depth"]#depth_range[:, 1][:, None, None, None]  # rfn,1,1,1
    # near_inv = -1 / near
    # far_inv = -1 / far
    depth = torch.clamp(depth, min=1e-5)
    # depth = -1 / depth
    # depth = (depth - near_inv) / (far_inv - near_inv)
    depth = (depth-near)/(far - near)
    depth = torch.clamp(depth, min=0, max=1.0) #归一化
    #disparity
    return depth

class MonoUncertWrapper(nn.Module):
  """Uncertainty Wrapper
  """

  def __init__(self, args, model):
    super().__init__()
    self.cfg = args
    self.mono_net = model 
    # load_mvs_model(self.mvs_net, args["mvs_checkpoints_dir"])
    load_checkpoint(args["mono_checkpoints_dir"], self.mono_net, "model_state_dict")
    num_layers = model.num_layers
    for param in self.mono_net.parameters():
      param.requires_grad = False
    self.mono_net.eval()
    # self.uncertainty_out =
    in_dim = 64
    depth_dim = 32

    self.equi_dec_convs = OrderedDict()
    self.num_ch_enc = np.array([64, 64, 128, 256, 512])
    if num_layers > 34:
        self.num_ch_enc[1:] *= 4

    if num_layers < 18:
        self.num_ch_enc = np.array([16, 24, 32, 96, 320])

    # decoder
    self.num_ch_dec = np.array([16, 32, 64, 128, 256])


    self.equi_dec_convs["std_upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])

    self.equi_dec_convs["std_deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
    self.equi_dec_convs["std_upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

    self.equi_dec_convs["std_deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
    self.equi_dec_convs["std_upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

    self.equi_dec_convs["std_deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
    self.equi_dec_convs["std_upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

    self.equi_dec_convs["std_deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
    self.equi_dec_convs["std_upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
    self.equi_dec_convs["std_deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])
    self.equi_dec_convs["std_conv_0"] = Conv3x3(self.num_ch_dec[0], 1)

    self.std_equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
    self.std_equi_decoder = erp_convert(self.std_equi_decoder)


  def forward(self, equi, cube, dnet=False, uncert_out=False):
      outputs  = self.mono_net(equi, cube, dnet=True, uncert_out=True)
      fused_feat4, fused_feat3, fused_feat2, fused_feat1, fused_feat0 = \
        outputs["fused_feat4"], outputs["fused_feat3"], outputs["fused_feat2"], outputs["fused_feat1"], outputs["fused_feat0"]

      equi_x = upsample(self.equi_dec_convs["std_upconv_5"](fused_feat4))
      equi_x = torch.cat([equi_x, fused_feat3], 1)
      equi_x = self.equi_dec_convs["std_deconv_4"](equi_x)
      equi_x = upsample(self.equi_dec_convs["std_upconv_4"](equi_x))

      equi_x = torch.cat([equi_x, fused_feat2], 1)
      equi_x = self.equi_dec_convs["std_deconv_3"](equi_x)
      equi_x = upsample(self.equi_dec_convs["std_upconv_3"](equi_x))

      equi_x = torch.cat([equi_x, fused_feat1], 1)
      equi_x = self.equi_dec_convs["std_deconv_2"](equi_x)
      equi_x = upsample(self.equi_dec_convs["std_upconv_2"](equi_x))


      equi_x = torch.cat([equi_x, fused_feat0], 1)
      equi_x = self.equi_dec_convs["std_deconv_1"](equi_x)
      
      equi_x = upsample(self.equi_dec_convs["std_upconv_1"](equi_x))
      equi_x = self.equi_dec_convs["std_deconv_0"](equi_x)
      mono_std = self.equi_dec_convs["std_conv_0"](equi_x)
      mono_std = F.softplus(mono_std, beta=20)

      outputs["mono_std"] = mono_std
      return outputs #mono_depth, mono_feat, mono_std

  # # def forward(self, mid_outputs, depth):
  # def estimate_depth_using_cost_volume(self, panos, rots, trans,
  #                                 min_depth=2, max_depth=100):
  #   ret_data = self.mvs_net.estimate_depth_using_cost_volume(panos, rots, trans, min_depth=min_depth, max_depth=max_depth)
  #   depth = ret_data["depth"].permute((0, 3, 1, 2))
  #   cost_reg = ret_data["cost_reg"].permute((0, 3, 1, 2))
  #   # print('depth.shape:', depth.shape)
  #   # print("cost_reg.shape:", cost_reg.shape)

  #   # import ipdb;ipdb.set_trace()
  #   # rgb_ = np.uint8(panos[0, 1].data.cpu().numpy()*255)

  #   # depth_ = depth[0, 0]

  #   # import os
  #   # import cv2
  #   # os.makedirs("./uncert_debug", exist_ok=True)
  #   # cv2.imwrite("./uncert_debug/rgb.jpg", rgb_)
  #   # def depth_norm(depth_np):
  #   #   d_min = depth_np.min()
  #   #   d_max = depth_np.max()
  #   #   d_norm = (depth_np-d_min)/(d_max-d_min)
  #   #   d_gray = np.uint8(d_norm*255)
  #   #   d_rgb = cv2.applyColorMap(d_gray, cv2.COLORMAP_JET)
  #   #   return d_rgb    
  #   # # import ipdb;ipdb.set_trace()
  #   # d_rgb = depth_norm(depth_.data.cpu().numpy())
  #   # cv2.imwrite("./uncert_debug/d_rgb.jpg", d_rgb)



  #   # import ipdb;ipdb.set_trace()
  #   depth = extract_depth_for_init_impl(self.cfg,  depth)
  #   depth = nn.functional.interpolate(
  #     depth,
  #     scale_factor=0.25,
  #     mode="bilinear",
  #     align_corners=False
  #   )#.permute((0, 2, 3, 1))
  #   # import ipdb;ipdb.set_trace()

  #   volume_feats = self.volume_conv2d(cost_reg)
  #   depth_feats = self.depth_conv(depth)
  #   volume_feats = torch.cat([volume_feats, depth_feats],1)
  #   var =  self.out_conv(volume_feats) #
  #   var = F.elu(var) + 1.0 + 1e-10
  #   ret_data["var"] = var
  #   # print("var.min():",var.min())
  #   ret_data["pred_final"] = torch.cat([ret_data["depth"].permute((0, 3, 1, 2)), var], dim=1)
  #   return ret_data

  # def activation_G(self, out):
  #   # mu, var = torch.split(out, 1, dim=1)
  #   var = F.elu(var) + 1.0 + 1e-10
  #   # out = torch.cat([mu, var], dim=1)  # (N, 2, H, W)
  #   return out

  def get_total_params(self):
    """Gets the total number of parameters.

    Returns:
      int total number of parameters.
    """
    # unet_params = my_torch_helpers.total_params(self.unet)
    # unet3d_params = my_torch_helpers.total_params(self.unet3d)
    # decoders1_params = my_torch_helpers.total_params(self.decoders1)
    # decoders2_params = my_torch_helpers.total_params(self.decoders2)
    # total_params = unet_params + unet3d_params + decoders1_params + decoders2_params
    # if self.args[""]
    mvsnet_params = my_torch_helpers.total_params(self.mono_net)
    # depth_conv_params = my_torch_helpers.total_params(self.depth_conv)
    # volume_conv2d_params = my_torch_helpers.total_params(self.volume_conv2d)
    # out_conv_params = my_torch_helpers.total_params(self.out_conv)
    # args = self.args
    total_params = mvsnet_params # + depth_conv_params + volume_conv2d_params + out_conv_params
    # if args["contain_dnet"]:
    #   total_params += my_torch_helpers.total_params(self.d_net)
    return total_params


