import torch.nn as nn
import os
import torch
from network.ops import ResidualBlock, conv3x3, conv1x1
from models.common_blocks import Upscale
from helpers import my_torch_helpers
import numpy as np
from network.omni_mvsnet.pipeline3_model import load_checkpoint
import torch.nn.functional as F
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

class UncertWrapper(nn.Module):
  """Uncertainty Wrapper
  """

  def __init__(self, args, model):
    super().__init__()
    self.cfg = args
    self.mvs_net = model 
    # load_mvs_model(self.mvs_net, args["mvs_checkpoints_dir"])
    load_checkpoint(args["mvs_checkpoints_dir"], self.mvs_net, "model_state_dict")    
    for param in self.mvs_net.parameters():
      param.requires_grad = False
    self.mvs_net.eval()
    # self.uncertainty_out =
    in_dim = 64
    depth_dim = 32
    # if not args["contain_dnet"]: #debug
    #   use_wrap_padding = False #args["use_wrap_padding"]
    # else:
    # if not args["contain_dnet"] and args["use_depth_sampling"]:
    #   use_wrap_padding = False #args["use_wrap_padding"]

    # else:
    use_wrap_padding = True #args["use_wrap_padding"]

    self.volume_conv2d = nn.Sequential(
      conv3x3(64, 32, use_wrap_padding=use_wrap_padding),
      ResidualBlock(32, 32, use_wrap_padding=use_wrap_padding),
      conv1x1(32, 32, use_wrap_padding=use_wrap_padding),
    ).cuda()

    # norm_layer = lambda dim: nn.InstanceNorm2d(dim, track_running_stats=False, affine=True)
    self.depth_conv = nn.Sequential(
      conv3x3(1, depth_dim, use_wrap_padding=use_wrap_padding),
      ResidualBlock(depth_dim, depth_dim, use_wrap_padding=use_wrap_padding),
      conv1x1(depth_dim, depth_dim, use_wrap_padding=use_wrap_padding)
    ).cuda()

    self.out_conv = nn.Sequential(
        conv3x3(in_dim, 32, use_wrap_padding=use_wrap_padding),
        ResidualBlock(32, 32, use_wrap_padding=use_wrap_padding),
        Upscale(),
        conv3x3(32, 16, use_wrap_padding=use_wrap_padding),
        Upscale(),
        conv1x1(16, 1, use_wrap_padding=use_wrap_padding)

    ).cuda()

  # def forward(self, mid_outputs, depth):
  def estimate_depth_using_cost_volume(self, panos, rots, trans,
                                  min_depth=2, max_depth=100):
    ret_data = self.mvs_net.estimate_depth_using_cost_volume(panos, rots, trans, min_depth=min_depth, max_depth=max_depth)
    depth = ret_data["depth"].permute((0, 3, 1, 2))
    cost_reg = ret_data["cost_reg"].permute((0, 3, 1, 2))
    # print('depth.shape:', depth.shape)
    # print("cost_reg.shape:", cost_reg.shape)

    # import ipdb;ipdb.set_trace()
    # rgb_ = np.uint8(panos[0, 1].data.cpu().numpy()*255)

    # depth_ = depth[0, 0]

    # import os
    # import cv2
    # os.makedirs("./uncert_debug", exist_ok=True)
    # cv2.imwrite("./uncert_debug/rgb.jpg", rgb_)
    # def depth_norm(depth_np):
    #   d_min = depth_np.min()
    #   d_max = depth_np.max()
    #   d_norm = (depth_np-d_min)/(d_max-d_min)
    #   d_gray = np.uint8(d_norm*255)
    #   d_rgb = cv2.applyColorMap(d_gray, cv2.COLORMAP_JET)
    #   return d_rgb    
    # # import ipdb;ipdb.set_trace()
    # d_rgb = depth_norm(depth_.data.cpu().numpy())
    # cv2.imwrite("./uncert_debug/d_rgb.jpg", d_rgb)



    # import ipdb;ipdb.set_trace()
    depth = extract_depth_for_init_impl(self.cfg,  depth)
    depth = nn.functional.interpolate(
      depth,
      scale_factor=0.25,
      mode="bilinear",
      align_corners=False
    )#.permute((0, 2, 3, 1))
    # import ipdb;ipdb.set_trace()

    volume_feats = self.volume_conv2d(cost_reg)
    depth_feats = self.depth_conv(depth)
    volume_feats = torch.cat([volume_feats, depth_feats],1)
    uncert =  self.out_conv(volume_feats) #
    if "new_uncert_tune" in self.cfg and self.cfg["new_uncert_tune"]:
      # import ipdb;ipdb.set_trace()
      var = F.softplus(uncert, beta=20).pow(2)#uncert = std
    else:
      var = F.elu(uncert) + 1.0 + 1e-10#uncert = var
    ret_data["var"] = var
    # print("var.min():",var.min())
    ret_data["pred_final"] = torch.cat([ret_data["depth"].permute((0, 3, 1, 2)), var], dim=1)
    return ret_data

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
    mvsnet_params = my_torch_helpers.total_params(self.mvs_net)
    depth_conv_params = my_torch_helpers.total_params(self.depth_conv)
    volume_conv2d_params = my_torch_helpers.total_params(self.volume_conv2d)
    out_conv_params = my_torch_helpers.total_params(self.out_conv)
    # args = self.args
    total_params = mvsnet_params + depth_conv_params + volume_conv2d_params + out_conv_params
    # if args["contain_dnet"]:
    #   total_params += my_torch_helpers.total_params(self.d_net)
    return total_params


