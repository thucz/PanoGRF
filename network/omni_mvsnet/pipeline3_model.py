# Lint as: python3
"""Encapsulates the full pipeline as a single torch module.
"""

import os
import sys
from einops import reduce
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from helpers import my_helpers
from helpers import my_torch_helpers

from models.common_blocks import (ConvBlock, Conv3DBlock, Conv3DBlockv2,
                                  ConvBlock2, UNet2)

from models.test_models import initialize_cost_volume_network
import math
import sys
sys.path.append("./UniFuse-Unidirectional-Fusion/UniFuse")
from datasets.util import Equirec2Cube
# from networks import UniFuse, Equi
# from networks.convert_module import erp_convert
# from networks.layers import Conv3x3


from models.spherical_cost_volume import calculate_cost_volume_erp
from models.spherical_cost_volume_mv import calculate_cost_volume_erp_multiview


# from models.inpainting_unet import InpaintNet
# from renderer.SpherePointsRasterizer import SpherePointsRasterizer
# from renderer.SpherePointsRenderer import SpherePointsRenderer

# Monodepth2 Imports
# sys.path.append('monodepth2')
# from monodepth2 import networks
# from monodepth2.layers import disp_to_depth, rot_from_axisangle

def input_normalize_depth(args, mu):
  return (mu - args["min_depth"])/(args["max_depth"]-args["min_depth"])

# load checkpoint
def load_checkpoint(fpath, model, model_name):
    ckpt = torch.load(fpath, map_location='cpu')
    if model_name in ckpt:
        ckpt = ckpt[model_name]
    
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v
   
    model.load_state_dict(load_dict)
    return model

class FullPipeline(nn.Module):
  """Full pipeline 3.
  """

  def __init__(self,args,
               device=torch.device("cuda"),
               monodepth_model="mono+stereo_1024x320",
               raster_resolution=256,
               width=256,
               height=256,
               layers=7,
               min_depth=0.1,
               max_depth=100.0,
               verbose=True,
               point_radius=0.004,
               points_per_pixel=8,
               depth_input_images=1,
               depth_output_channels=1,
               include_inpaintnet=False,
               include_poseestimator=False,
               input_uv=False,
               interpolation_mode="bilinear",
               cost_volume=False,
               use_v_input=False,
               depth_type="one_over",
               size=4,
               out_channels=1,
               **kwargs):
    """Initializes the pipeline.

    Args:
      device: Cuda device.
      monodepth_model: Name of the monodepth2 model. None for custom UNet.
      raster_resolution: Resolution of point cloud rendering.
      width: Width for custom unet.
      height: Height for custom unet.
      layers: Layers for the custom unet.
      verbose: Print debug statements.
      min_depth: Minimum depth.
      max_depth: Maximum depth.
      point_radius: Radius for point cloud renderer.
      points_per_pixel: Pointer per pixel for point cloud renderer.
      depth_input_images: Number of input images for the depth.
      depth_output_channels: Number of output images for the depth.
      include_inpaintnet: Include inpainting network.
      include_poseestimator: Include pose estimation.
      input_uv: Input UVs.
      interpolation_mode: Either bilinear or nearest.
      cost_volume: Use cost volume for depth estimation (requires 2 input images).
      use_v_input: Input v of uv coordinates into each conv block.
      **kwargs:
    """
    super().__init__()
    print("args:", args)
    self.args = args
    self.contain_dnet = args["contain_dnet"]
    if args["mvs_uncertainty"]:
      out_channels = 2

    if self.contain_dnet:
      self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      from select_mononet import select_mono
      self.d_net = select_mono(args, mvsnet=True)

      if self.args["mono_net"] == "PanoFormer" and self.args["dnet"] :
        import sys
        sys.path.append("./PanoFormer/PanoFormer")
        from network.model import OutputProj
        embed_dim=32
        img_size=256
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=embed_dim, kernel_size=3, stride=1,
                                      input_resolution=(img_size, img_size * 2))


      self.d_net.to(args["device"])
      if args["mono_uncert_tune"]:
        from network.omni_mvsnet.mono_uncert_wrapper import MonoUncertWrapper
        self.d_net = MonoUncertWrapper(args, self.d_net)

      self.d_net = load_checkpoint(args["DNET_ckpt"], self.d_net, "model_state_dict")#"dnet")

      for param in self.d_net.parameters():
          param.requires_grad = False
      self.d_net.eval()
      #revise
      self.e2c_mono = Equirec2Cube(args["mono_height"], args["mono_width"], args["mono_height"] // 2)

    self.e2c = Equirec2Cube(args["height"], args["width"], args["height"] // 2)    
    
    self.device = device
    self.verbose = verbose
    self.monodepth_model = monodepth_model
    self.width = width
    self.height = height
    self.layers = layers
    self.input_uv = input_uv
    self.min_depth = min_depth
    self.max_depth = max_depth
    self.interpolation_mode = interpolation_mode
    self.align_corners = False if interpolation_mode == "bilinear" else None
    self.using_monodepth_model = monodepth_model == "mono+stereo_1024x320"
    self.omega = None
    self.use_v_input = use_v_input
    self.depth_type = depth_type
    self.G = self.args["group_num"] if "group_num" in self.args and self.args["group_num"] > 1 else 1
    if self.contain_dnet:
      self.sampling_range = args["MAGNET_sampling_range"]        # beta in paper / defines the sampling range
      self.n_samples = args["MAGNET_num_samples"]                # N_s in paper / number of samples  
      self.k_list = self.depth_sampling()
      self.weighting = args["MAGNET_mvs_weighting"]
      if args["wo_hdh"]:
        cost_volume_channels = self.n_samples
      else:
        # cost_volume_channels = 64 #self.n_samples+32
        cost_volume_channels = 64 #59+5
    else:
      if "num_depth_hypothesis" in args:
        cost_volume_channels = args['num_depth_hypothesis']
      else:
        cost_volume_channels = 64
    self.cost_volume_channels = cost_volume_channels
          

    if isinstance(cost_volume, str):
      cost_volume = cost_volume.lower()
    if cost_volume == "true":
      cost_volume = "v1"
    self.use_cost_volume = cost_volume

    if cost_volume in ['v3', 'v3_cylindrical', 'v3_erp']:
      print('Using cost volume v3')
      if args["net"] == "OmniSyn":
        unet, unet3d, decoders1, decoders2 = self.initialize_cost_volume_network_v3(
          layers=layers, size=size,
          use_wrap_padding=True, use_v_input=use_v_input,
          out_channels=out_channels,
          cost_volume_channels=cost_volume_channels
        )
      
      else:
        unet, unet3d, decoders1, decoders2 = initialize_cost_volume_network(
          args,
          layers=layers, size=size,
          use_wrap_padding=True, use_v_input=use_v_input,
          out_channels=out_channels,
          cost_volume_channels=cost_volume_channels
        )
      self.unet = unet
      self.unet3d = unet3d
      self.decoders1 = decoders1
      self.decoders2 = decoders2
      if verbose:
        unet_params = my_torch_helpers.total_params(unet)
        cv_params = my_torch_helpers.total_params(unet3d)
        decoders2_params = my_torch_helpers.total_params(decoders2)
        total_params = unet_params + cv_params + decoders2_params
        print("Total depth params: %d" % (total_params,))

    # # estimate parameters
    # def estimate_depth_using_cost_volume(self, panos, rots, trans,
    #                                    min_depth=2, max_depth=100):

  def normalize_input(self, panos_small, panos_small_cube):      
      # inputs["normalized_rgb"] = normalize(rgb).unsqueeze(0)
      batch_size, seq_len = panos_small.shape[:2]
      normalized_equi = []
      normalized_cube = []
      for i in range(batch_size):
          pano_cube_seq = []
          pano_equi_seq = []
          for j in range(seq_len): 
              # cube = e2c_instance.run(panos_small[i, j].data.cpu().numpy())
              pano_equi_seq.append(self.normalize(panos_small[i, j]).data.cpu().numpy())
              pano_cube_seq.append(self.normalize(panos_small_cube[i, j]).data.cpu().numpy())
          normalized_equi.append(pano_equi_seq)
          normalized_cube.append(pano_cube_seq)

      # normalized_equi = np.array(normalized_equi)
      normalized_equi = torch.from_numpy(np.array(normalized_equi)).to(self.args["device"])
      normalized_cube = torch.from_numpy(np.array(normalized_cube)).to(self.args["device"])
      return normalized_equi, normalized_cube

  def normalize_erp_input(self, panos_small):      
      # inputs["normalized_rgb"] = normalize(rgb).unsqueeze(0)
      batch_size, seq_len = panos_small.shape[:2]
      normalized_equi = []
      # normalized_cube = []
      for i in range(batch_size):
          # pano_cube_seq = []
          pano_equi_seq = []
          for j in range(seq_len): 
              # cube = e2c_instance.run(panos_small[i, j].data.cpu().numpy())
              pano_equi_seq.append(self.normalize(panos_small[i, j]).data.cpu().numpy())
              # pano_cube_seq.append(self.normalize(panos_small_cube[i, j]).data.cpu().numpy())
          normalized_equi.append(pano_equi_seq)
          # normalized_cube.append(pano_cube_seq)

      # normalized_equi = np.array(normalized_equi)
      normalized_equi = torch.from_numpy(np.array(normalized_equi)).to(self.args["device"])
      # normalized_cube = torch.from_numpy(np.array(normalized_cube)).to(self.args["device"])
      return normalized_equi #, normalized_cube

  def e2c_process(self, panos_small, e2c_instance):      
      #panos_small = panos_small.reshape(batch_size, seq_len, height, width, 3)
      batch_size, seq_len = panos_small.shape[:2]
      pano_cube_all = []
      # import ipdb;ipdb.set_trace()
      for i in range(batch_size):
          # import ipdb;ipdb.set_trace()

          pano_cube_seq = []
          for j in range(seq_len):
              cube = e2c_instance.run(panos_small[i, j].data.cpu().numpy())
              pano_cube_seq.append(cube)
          # import ipdb;ipdb.set_trace()

          pano_cube_all.append(pano_cube_seq)
      pano_cube_all = np.array(pano_cube_all)
      pano_cube_all = torch.from_numpy(pano_cube_all).to(self.args["device"])
      return pano_cube_all
  def initialize_unet(self,
                      input_channels=3,
                      output_channels=1,
                      layers=7,
                      use_batchnorm=True,
                      use_wrap_padding=True,
                      gate=False):
    """Initializes a Unet.

    Args:
      input_channels: Input channels.
      output_channels: Output channels.
      use_batchnorm: Batch norm.
      use_wrap_padding:
      gate:

    Returns:

    """
    encoders = [
      ConvBlock(in_channels=input_channels,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
                upscale=False,
                use_batch_norm=use_batchnorm,
                use_wrap_padding=use_wrap_padding,
                gate=gate)
    ]
    decoders = [
      ConvBlock(in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                use_batch_norm=use_batchnorm,
                use_wrap_padding=use_wrap_padding,
                upscale=True)
    ]

    for i in range(1, layers):
      channels = 2 ** (i + 3)
      encoders.append(
        ConvBlock(in_channels=channels,
                  out_channels=2 * channels,
                  kernel_size=4,
                  stride=2,
                  padding=1,
                  use_batch_norm=use_batchnorm,
                  use_wrap_padding=use_wrap_padding,
                  upscale=False,
                  gate=gate))
      decoders.append(
        ConvBlock(in_channels=4 * channels,
                  out_channels=channels,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  use_batch_norm=use_batchnorm,
                  use_wrap_padding=use_wrap_padding,
                  upscale=True))

    encoders = nn.ModuleList(encoders)
    decoders = nn.ModuleList(decoders)
    final_conv = ConvBlock(in_channels=16,
                           out_channels=output_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           use_batch_norm=False,
                           use_wrap_padding=use_wrap_padding,
                           use_activation=False,
                           upscale=False)
    return encoders, decoders, final_conv

  def initialize_cost_volume_network_v3(self, layers=5, size=4,
                                        use_wrap_padding=True,
                                        use_v_input=False,
                                        out_channels=1, cost_volume_channels=64):
    """Initilizes a cost volume network.

    This initializes v2 of our cost volume network.
    The primary difference is that everything is a UNet now.
    Also we use UNet conv blocks which are conv-lrelu-conv-lrelu-pool.

    Args:
      layers: Layers.
      size: Size of channels.
      use_wrap_padding: Use wrap padding.
      use_v_input: Use v input.

    Returns:
      None.

    """
    encoders = [
      ConvBlock2(in_channels=3,
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 upscale=False,
                 use_wrap_padding=use_wrap_padding,
                 use_residual=False,
                 use_v_input=use_v_input)
    ]
    decoders = [None]

    for i in range(1, layers):
      channels = 2 ** (i + size - 1)
      encoders.append(
        ConvBlock2(in_channels=channels,
                   out_channels=2 * channels,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   use_wrap_padding=use_wrap_padding,
                   upscale=False,
                   use_residual=False,
                   use_v_input=use_v_input))
      if i > 1:
        decoders.append(
          ConvBlock2(in_channels=4 * channels,
                     out_channels=channels,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     use_wrap_padding=use_wrap_padding,
                     pooling=nn.Identity(),
                     upscale=False,
                     use_residual=False,
                     use_v_input=use_v_input))
      else:
        decoders.append(None)

    encoders.append(
      ConvBlock2(in_channels=2 ** (layers + size - 1),
                 out_channels=2 ** (layers + size),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 upscale=False,
                 use_residual=False,
                 pooling=nn.Identity(),
                 use_v_input=use_v_input))

    # The code below reduces the channel dimension to 1.
    cv_encoders = []
    cv_decoders = [
      Conv3DBlockv2(in_channels=2 ** (size + 3),
                    out_channels=1,
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                    use_batch_norm=False,
                    use_wrap_padding=use_wrap_padding,
                    pooling=nn.Identity(),
                    use_v_input=use_v_input)
    ]
    for i in range(0, 3):
      channels = 2 ** (i + size + 1)
      cv_encoders.append(
        Conv3DBlockv2(in_channels=channels,
                      out_channels=2 * channels,
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      use_batch_norm=False,
                      use_wrap_padding=use_wrap_padding,
                      use_v_input=use_v_input))
      if i > 0:
        cv_decoders.append(
          Conv3DBlockv2(in_channels=4 * channels,
                        out_channels=channels,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        use_batch_norm=False,
                        use_wrap_padding=use_wrap_padding,
                        pooling=nn.Identity(),
                        use_v_input=use_v_input))

    cv_encoders.append(
      Conv3DBlockv2(in_channels=2 ** (3 + size + 1),
                    out_channels=2 ** (3 + size + 2),
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                    use_batch_norm=False,
                    pooling=nn.Identity(),
                    use_wrap_padding=use_wrap_padding,
                    use_v_input=use_v_input))

    decoders1 = ConvBlock(
      cost_volume_channels,
      1,
      kernel_size=1,
      padding=0,
      stride=1,
      upscale=False,
      gate=False,
      use_wrap_padding=False,
      use_batch_norm=False,
      use_activation=False,
    )

    decoders2 = [
      ConvBlock2(in_channels=cost_volume_channels + 2 ** (size + 1),
                 out_channels=2 ** (size + 1),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 use_activation=True,
                 upscale=True,
                 use_residual=False,
                 pooling=False,
                 use_v_input=use_v_input),
      ConvBlock2(in_channels=2 ** (size + 1),
                 out_channels=2 ** size,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 use_activation=True,
                 upscale=True,
                 use_residual=False,
                 pooling=False,
                 use_v_input=use_v_input),
      ConvBlock2(in_channels=2 ** size,
                 out_channels=out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_wrap_padding=use_wrap_padding,
                 use_activation=False,
                 upscale=False,
                 use_residual=False,
                 pooling=False,
                 use_v_input=use_v_input),
    ]

    unet = UNet2(nn.ModuleList(encoders),
                 nn.ModuleList(decoders))
    unet3d = UNet2(nn.ModuleList(cv_encoders),
                   nn.ModuleList(cv_decoders),
                   interpolation="trilinear",
                   name="unet3d")
    decoders2 = nn.ModuleList(decoders2)
    return unet, unet3d, decoders1, decoders2

  def depth_sampling(self):
    from scipy.special import erf
    from scipy.stats import norm
    P_total = erf(self.sampling_range / np.sqrt(2))             # Probability covered by the sampling range
    idx_list = np.arange(0, self.n_samples + 1)
    p_list = (1 - P_total)/2 + ((idx_list/self.n_samples) * P_total)
    k_list = norm.ppf(p_list)
    k_list = (k_list[1:] + k_list[:-1])/2
    return list(k_list)

  def forward_thru_unet(self, input):
    """Forward pass through the UNet.

    Args:
      input: Input as channels first tensor.

    Returns:
      Output of the forward pass as a channels first tensor.

    """
    x = input
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
    return x

  def estimate_depth_using_cost_volume(self, panos, rots, trans,
                                       min_depth=2, max_depth=100):
    """Estimates depth using cost volume either v1 or v2.

    Args:
      panos: Panos.
      rots: Trans.
      trans: Rots.
      min_depth: Minimum depth.
      max_depth: Maximum depth.

    Returns:
      Predicted depth.
    """
    
    if self.use_cost_volume == "v3_erp":
      return self.estimate_depth_using_cost_volume_v3_erp(panos, rots, trans,
                                                          min_depth=min_depth,
                                                          max_depth=max_depth)
    return None


  def expand_sin(self, x):
    if self.args["with_sin"]:
      bs, _, height, width = x.shape
      sin_phi = torch.arange(0, height, dtype=torch.float32).cuda()
      sin_phi = torch.sin((sin_phi + 0.5) * math.pi / (height))
      sin_phi = sin_phi.view(1, 1, height, 1).expand(bs, 1, height, width)
      x = torch.cat([x, sin_phi],dim=1)  
    return x      

  def estimate_depth_using_cost_volume_v3_erp(self, panos, rots, trans,
                                              min_depth=2.0, max_depth=100):

    """Estimates depth of images[:, 1] using a cost volume.

    Args:
      panos: Tensor of shape (B, 2, H, W, 3)
      rots: Tensor of shape (B, 2, 3, 3)
      trans: Tensor of shape (B, 2, 3)

    Returns:
      Estimated depth.

    """

    # panos = panos.permute((0, 1, 3, 4, 2))
    args = self.args


    batch_size, seq_len, height, width, _ = panos.shape
    # print("height, width:", height, width)
    B = batch_size

    # trans_norm = torch.norm(trans[:, 0] - trans[:, 1], dim=1)
    
    # rectified_panos, rect_rots = self.rectify_images(panos, rots, trans,
    #                                                  rectify_vertical=False)

    
    # Change to channels first.
    panos_cf = panos.permute((0, 1, 4, 2, 3))
    if self.args["contain_dnet"]:
      #mono-net
      args = self.args
      mono_panos_cf = panos_cf.reshape(batch_size*seq_len, 3, args["height"], args["width"])
      mono_panos_cf = torch.nn.functional.interpolate(mono_panos_cf,
                                                (args["mono_height"], args["mono_width"]),
                                                mode=args["interpolation_mode"],
                                                align_corners=False)
      mono_panos_cf = mono_panos_cf.reshape(batch_size, seq_len, 3, args["mono_height"], args["mono_width"])
      mono_panos_cf = mono_panos_cf.permute((0, 1, 3, 4, 2))
      
      mono_panos_cube = self.e2c_process(mono_panos_cf, self.e2c_mono)

    if self.args["contain_dnet"]:
      with torch.no_grad():
        # D-Net forward pass
        # Change to channels first.
        mono_panos_cf = mono_panos_cf.permute((0, 1, 4, 2, 3))
        mono_panos_cube = mono_panos_cube.permute((0, 1, 4, 2, 3))
        
        if self.args["mono_net"]=="UniFuse":
          mono_erp_inputs, mono_cube_inputs = self.normalize_input(mono_panos_cf, mono_panos_cube)
          mono_erp_inputs = torch.cat((mono_erp_inputs[:, 1], mono_erp_inputs[:, 0]), dim=0)
          mono_cube_inputs = torch.cat((mono_cube_inputs[:, 1], mono_cube_inputs[:, 0]), dim=0)
          mono_outputs = self.d_net(mono_erp_inputs, mono_cube_inputs, dnet=True)
          
          x_d3 = mono_outputs["mono_feat"]
          if self.args["mono_uncertainty"]:
            mono_gmms = mono_outputs["pred"]
          else:
            mono_gmms = mono_outputs["pred_depth"]  
          mono_gmms_orig = mono_gmms.clone()
          mono_gmms = F.interpolate(mono_gmms, [height//4, width//4], mode='bilinear')
          x_d3 = F.interpolate(x_d3, [height//4, width//4], mode='bilinear')
          if self.args["mono_uncert_tune"]:
            mono_std = mono_outputs["mono_std"]
            mono_std = F.interpolate(mono_std, [height//4, width//4], mode='bilinear')
      
          mono_gmms = mono_gmms.detach()
          ref_gmms = mono_gmms[:B, ...]
          x_d3 = x_d3[:B, ...]
          nghbr_gmms = mono_gmms[B:, ...]
          if self.args["mono_uncert_tune"]:
            ref_sigma = mono_std[:B, ...]
        elif self.args["mono_net"] in ["ACDNet"]:
          # mono_panos_cf
          mono_erp_inputs = torch.cat((mono_panos_cf[:, 1], mono_panos_cf[:, 0]), dim=0)
          mono_outputs = self.d_net(mono_erp_inputs, dnet=self.args["dnet"]) 
          if self.args["dnet"]:
            x_d3 = mono_outputs["mono_feat"]
            x_d3 = F.interpolate(x_d3, [height//4, width//4], mode='bilinear')
            x_d3 = x_d3[:B, ...]

          mono_gmms = mono_outputs["depth"]          
          mono_gmms_orig = mono_gmms.clone()
          mono_gmms = F.interpolate(mono_gmms, [height//4, width//4], mode='bilinear')
  
          mono_gmms = mono_gmms.detach()
          ref_gmms = mono_gmms[:B, ...]
          nghbr_gmms = mono_gmms[B:, ...]
          
        elif self.args["mono_net"] in ["PanoFormer"]:
          mono_erp_inputs = self.normalize_erp_input(mono_panos_cf)          
          mono_erp_inputs = torch.cat((mono_erp_inputs[:, 1], mono_erp_inputs[:, 0]), dim=0)
          mono_outputs = self.d_net(mono_erp_inputs, dnet=self.args["dnet"])

          mono_gmms = mono_outputs["pred_depth"]
          if self.args["dnet"]:
            x_d3 = mono_outputs["mono_feat"]
            x_d3 = self.output_proj(x_d3)

            x_d3 = F.interpolate(x_d3, [height//4, width//4], mode='bilinear')

            x_d3 = x_d3[:B, ...]

          mono_gmms_orig = mono_gmms.clone()
          mono_gmms = F.interpolate(mono_gmms, [height//4, width//4], mode='bilinear')
          mono_gmms = mono_gmms.detach()
          ref_gmms = mono_gmms[:B, ...]
          nghbr_gmms = mono_gmms[B:, ...]
        else:
          raise Exception

    #   # pred_list = [ref_gmms]
      ref_mu = ref_gmms[:, :1, ...]
      if self.args["mono_uncertainty"]:
        ref_sigma = ref_gmms[:, 1:, ...]

    #   #todo
      # ref_sigma = (max_depth - min_depth)/self.n_samples
      if self.n_samples>0:
        if self.args["mono_uncertainty"] or args["mono_uncert_tune"]:
          if args["relaxation_factor"] in self.args:
            print("use relaxation_factor:", args["relaxation_factor"])
            
            depth_volume = [torch.clamp(ref_mu + ref_sigma * k * self.args["relaxation_factor"], min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
          else:
            depth_volume = [torch.clamp(ref_mu + torch.clamp(ref_sigma, min=args["basic_sigma"]) * k, min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
        else:
          depth_volume = [torch.clamp(ref_mu + k*self.args["fixed_sigma"], min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
        depth_volume = torch.cat(depth_volume, dim=1)                       # B, N_samples, H, W    



    image_features_0_cf = panos_cf[:, 0]
    image_features_1_cf = panos_cf[:, 1]
    # Pass through encoding UNet.
    if self.args["net"]=="OmniSyn":
      image_features_0_cf = self.unet(image_features_0_cf)
      image_features_1_cf = self.unet(image_features_1_cf)
    else:
      panos_cube = self.e2c_process(panos, self.e2c)    
      panos_cube = panos_cube.permute((0, 1, 4, 2, 3))
      image_features_0_cube=panos_cube[:, 0]
      image_features_1_cube=panos_cube[:, 1]
      if self.args["contain_dnet"] and self.args["change_input"]:
        ref_mu_orig = mono_gmms_orig[:B, :1, ...]
        other_mu_orig = mono_gmms_orig[B:, :1, ...]
        # depth_image = torch.cat([])
        depth_image_1 = input_normalize_depth(args, ref_mu_orig)
        depth_image_0 = input_normalize_depth(args, other_mu_orig)
        image_features_0_cf = self.unet(image_features_0_cf, image_features_0_cube, depth_image_0)
        image_features_1_cf = self.unet(image_features_1_cf, image_features_1_cube, depth_image_1)

      else:
        image_features_0_cf = self.unet(image_features_0_cf, image_features_0_cube)
        image_features_1_cf = self.unet(image_features_1_cf, image_features_1_cube)

    # Change to channels last.
    image_features_0 = image_features_0_cf.permute((0, 2, 3, 1))#B, H, W, 1
    image_features_1 = image_features_1_cf.permute((0, 2, 3, 1))#B, H, W, 1

    # Create cost volume.
    image_features = torch.stack((image_features_0, image_features_1), dim=1)
    
    # import pdb;pdb.set_trace()
    # cost_volume = self.calculate_cost_volume_erp(
    #   image_features,
    #   depths=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, 64),
    #   trans_norm=trans_norm,
    #   cost_type='abs_diff')
    if self.contain_dnet:
      thres = int(self.weighting.split('CW')[1])
      if args["wo_hdh"]:
        d_centers = None
        pass
        # depth_volume = torch.cat([depth_volume, d_centers], dim=1)      
      else:
        n_depths = self.cost_volume_channels - self.n_samples
        if args["use_depth_sampling"]:
          if args["revise_range"]:
            # [-fixed_sigma, + fixed_sigma]
            # d_centers = ref_mu + fixed_sigma * k
            fixed_dist = args["fixed_dist"]
            #B, 1, H, W
            d_min = torch.clamp(ref_mu - fixed_dist, min=args["min_depth"])
            d_max = torch.clamp(ref_mu + fixed_dist, max=args["max_depth"])
            # d_centers = [torch.clamp(ref_mu + ref_sigma * k * self.args["relaxation_factor"], min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
            # revise_min_depth =             
            # d_centers=torch.linspace(min_depth, max_depth, n_depths).reshape(1, n_depths, 1, 1).cuda()
            interval = (d_max-d_min)/(n_depths-1) # B, 1, H, W
            # start = d_min
            vals = torch.arange(0, n_depths-1).reshape(1, n_depths-1, 1, 1).cuda()#1, D-1, H, W
            adds = interval * vals #B, D-1, H, W            
            d_centers = torch.cat([d_min, d_min+adds], dim=1)
          else:
            d_centers=torch.linspace(min_depth, max_depth, n_depths).reshape(1, n_depths, 1, 1).cuda()          
        else:
          d_centers=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, n_depths).reshape(1, n_depths, 1, 1).cuda()
        
        
        # B, D, H, W = depth_volume.shape
        B, _, H, W = ref_mu.shape
        if args["revise_range"]:
          pass
        else:
          d_centers = d_centers.repeat(B, 1, H, W)                
        
        if self.n_samples>0:
          depth_volume = torch.cat([depth_volume, d_centers], dim=1)
        else:
          depth_volume = d_centers #torch.cat([depth_volume, d_centers], dim=1)
        depth_volume, _ = torch.sort(depth_volume, dim=1)#important!!! avoid overfit to mono depth hypotheses
    else:
      n_depths = self.cost_volume_channels
      if args["use_depth_sampling"]:
        d_centers=torch.linspace(min_depth, max_depth, n_depths)#.reshape(1, n_depths, 1, 1).cuda()
      else:
        d_centers=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, n_depths)#.reshape(1, n_depths, 1, 1).cuda()
    if self.contain_dnet:
      cost_volume = calculate_cost_volume_erp(self.args,
        image_features,
        depths=d_centers,
        depth_volume=depth_volume,
        trans = trans,
        rots = rots,
        cost_type='abs_diff',
        ref_gmms=ref_gmms,
        nghbr_gmms = nghbr_gmms,
        thres=thres)
    else:
      cost_volume = calculate_cost_volume_erp(self.args,
        image_features,
        depths=d_centers,
        trans = trans,
        rots = rots,
        cost_type='abs_diff')
    # Pass cost volume through Conv3D layers.
    # if self.args["wo3dcnn"]:
    #   cost_volume = self.unet3d(cost_volume)
    #   image_features = cost_volume #cost_volume[:, 0, :, :, :]

    # else:
    
    cost_volume = cost_volume.permute((0, 4, 1, 2, 3)) #B, D, H, W, F -> B, F, D, H, W

    if "group_wise" in self.args and self.args["group_wise"]:
    # if self.args["group"]
      C = cost_volume.shape[1]
      cost_volume = cost_volume.view(B, self.G, C//self.G, *cost_volume.shape[-3:])
      cost_volume = reduce(cost_volume, 'b g c d h w -> b g d h w', 'mean') # (B, G, D, h, w)
      # cost_volume = 
    cost_volume = self.unet3d(cost_volume)
    # Now the length dimension is now the channel dimension.
    image_features = cost_volume[:, 0, :, :, :]
    
    cost_reg = image_features #B, D, H, W
    # if args["mvs_uncert"]:

    #   raw_image_features_d1 = self.decoders1(torch.cat(
    #         (cost_volume[:, 0, :, :, :], image_features_1_cf),
    #         dim=1))
  # else: 
    raw_image_features_d1 = self.decoders1(image_features)
    raw_image_features_d1 = nn.functional.interpolate(
      raw_image_features_d1,
      scale_factor=4,
      mode="bilinear",
      align_corners=False
    ).permute((0, 2, 3, 1))

    # Here the cost volume already takes into account the translation.
    if self.args["out_type"]=='disparity':
      eps=1e-10
      rectified_depth_d1 = 1.0 / (torch.clamp(raw_image_features_d1, min=0)+eps)
    elif self.args["out_type"] == 'depth':
      rectified_depth_d1 = torch.clamp(raw_image_features_d1, min=0)

    # Now the length dimension is now the channel dimension.
    # image_features = cost_volume[:, 0, :, :, :]
    if self.args["wo_mono_feat"]:

      image_features = cost_volume[:, 0, ...]
    else:
      if self.args["contain_dnet"] and self.args["mono_net"]=="UniFuse" :
        image_features = torch.cat(
          (cost_volume[:, 0, :, :, :], x_d3),
          dim=1)
      elif self.args["contain_dnet"] and self.args["mono_net"] in ["ACDNet", "PanoFormer"] and self.args["dnet"]:
        image_features = torch.cat(
          (cost_volume[:, 0, :, :, :], x_d3),
          dim=1)
      else:  
        image_features = torch.cat(
          (cost_volume[:, 0, :, :, :], image_features_1_cf),
          dim=1)
      if self.args["with_sin"]:
        image_features = self.expand_sin(image_features)

    mid_outputs = {
      "cost_volume": cost_volume,
      "image_features_1": image_features,
    }
    #mono-stereo fuse layer:
    for i in range(len(self.decoders2)):
      image_features, _ = self.decoders2[i](image_features)
    
    # mid_outputs["image_features_2"] = image_features

    assert torch.isfinite(image_features).all(), "Nan in image features"
    # Here the cost volume already takes into account the translation.    
    if self.args["mvs_uncertainty"]:
      # print("image_features[:, :1].shape:", image_features[:, :1].shape)
      pred = self.activation_G(image_features) #out_type: depth
      pred_d = pred[:, :1, ...]
      if self.args["out_type"] == "disparity":
        eps=1e-10
        rectified_depth = 1.0/ (torch.clamp(pred_d, min=0)+eps)
      elif self.args["out_type"] == "depth":
        rectified_depth = torch.clamp(pred_d, min=0)
      pred_final = torch.cat([rectified_depth, pred[:, 1:, ...]], dim=1)
    else:
      if self.args["out_type"] == "disparity":
        eps=1e-10
        rectified_depth = 1.0/ (torch.clamp(image_features[:, :1], min=0)+eps)
      elif self.args["out_type"] == "depth":
        rectified_depth = torch.clamp(image_features[:, :1], min=0)

    rectified_depth = rectified_depth.permute((0, 2, 3, 1))
    depth = rectified_depth
    cost_reg = cost_reg.permute((0, 2, 3, 1))
    return_data = {
      "depth": depth,
      "rectified_depth_d1": rectified_depth_d1,
      "rectified_depth": rectified_depth,
      "raw_image_features": image_features.permute((0, 2, 3, 1)),
      "raw_image_features_d1": raw_image_features_d1,
      "cost_reg": cost_reg,
      "mid_outputs": mid_outputs
    }
    if self.args["mvs_uncertainty"]:
      return_data["pred_final"] = pred_final
    if self.args["contain_dnet"]:
      ref_mu = ref_mu.permute((0, 2, 3, 1))
      ref_mu = ref_mu.permute((0, 3, 1, 2))
      return_data["mono_depth_ref"] = ref_mu
    return return_data

  def estimate_depth_using_cost_volume_v3_erp_multiview(self, panos, rots, trans, curr_idx,
                                              min_depth=2.0, max_depth=100):

    """Estimates depth of images[:, 1] using a cost volume.

    Args:
      panos: Tensor of shape (B, 2, H, W, 3)
      rots: Tensor of shape (B, 2, 3, 3)
      trans: Tensor of shape (B, 2, 3)

    Returns:
      Estimated depth.

    """

    # panos = panos.permute((0, 1, 3, 4, 2))
    args = self.args
    batch_size, seq_len, height, width, _ = panos.shape
    # print("height, width:", height, width)
    B = batch_size
    # trans_norm = torch.norm(trans[:, 0] - trans[:, 1], dim=1)    
    # rectified_panos, rect_rots = self.rectify_images(panos, rots, trans,
    #                                                  rectify_vertical=False)
    
    # Change to channels first.
    panos_cf = panos.permute((0, 1, 4, 2, 3))
    if self.args["contain_dnet"]:
      #mono-net
      args = self.args
      mono_panos_cf = panos_cf.reshape(batch_size*seq_len, 3, args["height"], args["width"])
      mono_panos_cf = torch.nn.functional.interpolate(mono_panos_cf,
                                                (args["mono_height"], args["mono_width"]),
                                                mode=args["interpolation_mode"],
                                                align_corners=False)
      mono_panos_cf = mono_panos_cf.reshape(batch_size, seq_len, 3, args["mono_height"], args["mono_width"])
      mono_panos_cf = mono_panos_cf.permute((0, 1, 3, 4, 2))
      
      mono_panos_cube = self.e2c_process(mono_panos_cf, self.e2c_mono)

    if self.args["contain_dnet"]:
      with torch.no_grad():
        # D-Net forward pass
        # Change to channels first.
        mono_panos_cf = mono_panos_cf.permute((0, 1, 4, 2, 3))
        mono_panos_cube = mono_panos_cube.permute((0, 1, 4, 2, 3))
        if self.args["mono_net"]=="UniFuse":
          mono_erp_inputs, mono_cube_inputs = self.normalize_input(mono_panos_cf, mono_panos_cube)
          # mono_erp_inputs = torch.cat((mono_erp_inputs[:, 1], mono_erp_inputs[:, 0]), dim=0)
          # mono_cube_inputs = torch.cat((mono_cube_inputs[:, 1], mono_cube_inputs[:, 0]), dim=0)
          # mono_erp_inputs = torch.cat((mono_erp_inputs[:, 1], mono_erp_inputs[:, 0]), dim=0)
          # mono_cube_inputs = torch.cat((mono_cube_inputs[:, 1], mono_cube_inputs[:, 0]), dim=0)
          mono_outputs = self.d_net(mono_erp_inputs[:, curr_idx], mono_cube_inputs[:, curr_idx], dnet=True)
          x_d3 = mono_outputs["mono_feat"]
          if self.args["mono_uncertainty"]:
            mono_gmms = mono_outputs["pred"]
          else:
            mono_gmms = mono_outputs["pred_depth"]  
          mono_gmms_orig = mono_gmms.clone()
          mono_gmms = F.interpolate(mono_gmms, [height//4, width//4], mode='bilinear')
          x_d3 = F.interpolate(x_d3, [height//4, width//4], mode='bilinear')
          if self.args["mono_uncert_tune"]:
            mono_std = mono_outputs["mono_std"]
            mono_std = F.interpolate(mono_std, [height//4, width//4], mode='bilinear')
      
          mono_gmms = mono_gmms.detach()
          ref_gmms = mono_gmms[:B, ...]
          x_d3 = x_d3[:B, ...]
          # nghbr_gmms = mono_gmms[B:, ...]
          if self.args["mono_uncert_tune"]:
            ref_sigma = mono_std[:B, ...]
        # elif self.args["mono_net"] in ["ACDNet"]:
        #   # mono_panos_cf
        #   mono_erp_inputs = torch.cat((mono_panos_cf[:, 1], mono_panos_cf[:, 0]), dim=0)
        #   mono_outputs = self.d_net(mono_erp_inputs, dnet=self.args["dnet"]) 
        #   if self.args["dnet"]:
        #     x_d3 = mono_outputs["mono_feat"]
        #     x_d3 = F.interpolate(x_d3, [height//4, width//4], mode='bilinear')
        #     x_d3 = x_d3[:B, ...]

        #   mono_gmms = mono_outputs["depth"]          
        #   mono_gmms_orig = mono_gmms.clone()
        #   mono_gmms = F.interpolate(mono_gmms, [height//4, width//4], mode='bilinear')
  
        #   mono_gmms = mono_gmms.detach()
        #   ref_gmms = mono_gmms[:B, ...]
        #   nghbr_gmms = mono_gmms[B:, ...]
          
        # elif self.args["mono_net"] in ["PanoFormer"]:
        #   mono_erp_inputs = self.normalize_erp_input(mono_panos_cf)          
        #   mono_erp_inputs = torch.cat((mono_erp_inputs[:, 1], mono_erp_inputs[:, 0]), dim=0)
        #   mono_outputs = self.d_net(mono_erp_inputs, dnet=self.args["dnet"])

        #   mono_gmms = mono_outputs["pred_depth"]
        #   if self.args["dnet"]:
        #     x_d3 = mono_outputs["mono_feat"]
        #     x_d3 = self.output_proj(x_d3)

        #     x_d3 = F.interpolate(x_d3, [height//4, width//4], mode='bilinear')

        #     x_d3 = x_d3[:B, ...]

        #   mono_gmms_orig = mono_gmms.clone()
        #   mono_gmms = F.interpolate(mono_gmms, [height//4, width//4], mode='bilinear')
        #   mono_gmms = mono_gmms.detach()
        #   ref_gmms = mono_gmms[:B, ...]
        #   nghbr_gmms = mono_gmms[B:, ...]
        else:
          raise Exception

    #   # pred_list = [ref_gmms]
      ref_mu = ref_gmms[:, :1, ...]
      if self.args["mono_uncertainty"]:
        ref_sigma = ref_gmms[:, 1:, ...]
    #   #todo
      # ref_sigma = (max_depth - min_depth)/self.n_samples
      if self.n_samples>0:
        if self.args["mono_uncertainty"] or args["mono_uncert_tune"]:
          if args["relaxation_factor"] in self.args:
            print("use relaxation_factor:", args["relaxation_factor"])
            
            depth_volume = [torch.clamp(ref_mu + ref_sigma * k * self.args["relaxation_factor"], min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
          else:
            depth_volume = [torch.clamp(ref_mu + torch.clamp(ref_sigma, min=args["basic_sigma"]) * k, min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
        else:
          depth_volume = [torch.clamp(ref_mu + k*self.args["fixed_sigma"], min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
        depth_volume = torch.cat(depth_volume, dim=1)                       # B, N_samples, H, W    


    # image_features_cf = panos_cf
    # image_features_0_cf = panos_cf[:, 0]
    # image_features_1_cf = panos_cf[:, 1]
    # Pass through encoding UNet.
    # if self.args["net"]=="OmniSyn":
    #   image_features_0_cf = self.unet(image_features_0_cf)
    #   image_features_1_cf = self.unet(image_features_1_cf)
    # else:
      panos_cube = self.e2c_process(panos, self.e2c)    
      panos_cube = panos_cube.permute((0, 1, 4, 2, 3))
      # image_features_0_cube=panos_cube[:, 0]
      # image_features_1_cube=panos_cube[:, 1]
      # if self.args["contain_dnet"] and self.args["change_input"]:
      #   ref_mu_orig = mono_gmms_orig[:B, :1, ...]
      #   other_mu_orig = mono_gmms_orig[B:, :1, ...]
      #   # depth_image = torch.cat([])
      #   depth_image_1 = input_normalize_depth(args, ref_mu_orig)
      #   depth_image_0 = input_normalize_depth(args, other_mu_orig)
      #   image_features_0_cf = self.unet(image_features_0_cf, image_features_0_cube, depth_image_0)
      #   image_features_1_cf = self.unet(image_features_1_cf, image_features_1_cube, depth_image_1)

      # else:
        # image_features_0_cf = self.unet(image_features_0_cf, image_features_0_cube)
        # image_features_1_cf = self.unet(image_features_1_cf, image_features_1_cube)
      # panos_cf.shape
      batch_size, seq_len, channel, height, width = panos_cf.shape
      image_features_cf = self.unet(panos_cf.reshape(batch_size*seq_len, channel, height, width))
    _, f_channel, f_height, f_width = image_features_cf.shape
    image_features = image_features_cf.reshape(batch_size, seq_len, f_channel, f_height, f_width)
    image_features = image_features.permute((0, 1, 3, 4, 2))# Change to channels last.

    # Create cost volume.
    # image_features = torch.stack((image_features_0, image_features_1), dim=1) #B, 2, H, W, F   
    
    # import pdb;pdb.set_trace()
    # cost_volume = self.calculate_cost_volume_erp(
    #   image_features,
    #   depths=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, 64),
    #   trans_norm=trans_norm,
    #   cost_type='abs_diff')
    if self.contain_dnet:
      thres = int(self.weighting.split('CW')[1])
      if args["wo_hdh"]:
        d_centers = None
        pass
        # depth_volume = torch.cat([depth_volume, d_centers], dim=1)      
      else:
        n_depths = self.cost_volume_channels - self.n_samples
        if args["use_depth_sampling"]:
          if args["revise_range"]:
            # [-fixed_sigma, + fixed_sigma]
            # d_centers = ref_mu + fixed_sigma * k
            fixed_dist = args["fixed_dist"]
            #B, 1, H, W
            d_min = torch.clamp(ref_mu - fixed_dist, min=args["min_depth"])
            d_max = torch.clamp(ref_mu + fixed_dist, max=args["max_depth"])
            # d_centers = [torch.clamp(ref_mu + ref_sigma * k * self.args["relaxation_factor"], min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
            # revise_min_depth =             
            # d_centers=torch.linspace(min_depth, max_depth, n_depths).reshape(1, n_depths, 1, 1).cuda()
            interval = (d_max-d_min)/(n_depths-1) # B, 1, H, W
            # start = d_min
            vals = torch.arange(0, n_depths-1).reshape(1, n_depths-1, 1, 1).cuda()#1, D-1, H, W
            adds = interval * vals #B, D-1, H, W            
            d_centers = torch.cat([d_min, d_min+adds], dim=1)
          else:
            d_centers=torch.linspace(min_depth, max_depth, n_depths).reshape(1, n_depths, 1, 1).cuda()          
        else:
          d_centers=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, n_depths).reshape(1, n_depths, 1, 1).cuda()
        
        
        # B, D, H, W = depth_volume.shape
        B, _, H, W = ref_mu.shape
        if args["revise_range"]:
          pass
        else:
          d_centers = d_centers.repeat(B, 1, H, W)                
        
        if self.n_samples>0:
          depth_volume = torch.cat([depth_volume, d_centers], dim=1)
        else:
          depth_volume = d_centers #torch.cat([depth_volume, d_centers], dim=1)
        depth_volume, _ = torch.sort(depth_volume, dim=1)#important!!! avoid overfit to mono depth hypotheses
    else:
      n_depths = self.cost_volume_channels
      if args["use_depth_sampling"]:
        d_centers=torch.linspace(min_depth, max_depth, n_depths)#.reshape(1, n_depths, 1, 1).cuda()
      else:
        d_centers=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, n_depths)#.reshape(1, n_depths, 1, 1).cuda()
    if self.contain_dnet:
      cost_volume = calculate_cost_volume_erp_multiview(self.args,
        image_features,
        depths=d_centers,
        depth_volume=depth_volume,
        trans = trans,
        rots = rots,
        cost_type='abs_diff',
        ref_gmms=ref_gmms,
        # nghbr_gmms = nghbr_gmms,
        thres=thres,
        curr_idx=curr_idx)
    else:
      cost_volume = calculate_cost_volume_erp_multiview(self.args,
        image_features,
        depths=d_centers,
        trans = trans,
        rots = rots,
        cost_type='abs_diff',
        curr_idx=curr_idx)
    # Pass cost volume through Conv3D layers.
    # if self.args["wo3dcnn"]:
    #   cost_volume = self.unet3d(cost_volume)
    #   image_features = cost_volume #cost_volume[:, 0, :, :, :]

    # else:
    
    cost_volume = cost_volume.permute((0, 4, 1, 2, 3)) #B, D, H, W, F -> B, F, D, H, W

    if "group_wise" in self.args and self.args["group_wise"]:
    # if self.args["group"]
      C = cost_volume.shape[1]
      cost_volume = cost_volume.view(B, self.G, C//self.G, *cost_volume.shape[-3:])
      cost_volume = reduce(cost_volume, 'b g c d h w -> b g d h w', 'mean') # (B, G, D, h, w)
      # cost_volume = 
    cost_volume = self.unet3d(cost_volume)
    # Now the length dimension is now the channel dimension.
    image_features = cost_volume[:, 0, :, :, :]
    
    cost_reg = image_features #B, D, H, W
    # if args["mvs_uncert"]:

    #   raw_image_features_d1 = self.decoders1(torch.cat(
    #         (cost_volume[:, 0, :, :, :], image_features_1_cf),
    #         dim=1))
  # else: 
    raw_image_features_d1 = self.decoders1(image_features)
    raw_image_features_d1 = nn.functional.interpolate(
      raw_image_features_d1,
      scale_factor=4,
      mode="bilinear",
      align_corners=False
    ).permute((0, 2, 3, 1))

    # Here the cost volume already takes into account the translation.
    if self.args["out_type"]=='disparity':
      eps=1e-10
      rectified_depth_d1 = 1.0 / (torch.clamp(raw_image_features_d1, min=0)+eps)
    elif self.args["out_type"] == 'depth':
      rectified_depth_d1 = torch.clamp(raw_image_features_d1, min=0)
    
    # Now the length dimension is now the channel dimension.
    # image_features = cost_volume[:, 0, :, :, :]
    if self.args["wo_mono_feat"]:
      image_features = cost_volume[:, 0, ...]
    else:
      if self.args["contain_dnet"] and self.args["mono_net"]=="UniFuse" :
        image_features = torch.cat(
          (cost_volume[:, 0, :, :, :], x_d3),
          dim=1)
      elif self.args["contain_dnet"] and self.args["mono_net"] in ["ACDNet", "PanoFormer"] and self.args["dnet"]:
        image_features = torch.cat(
          (cost_volume[:, 0, :, :, :], x_d3),
          dim=1)
      else:  
        image_features = torch.cat(
          (cost_volume[:, 0, :, :, :], image_features_1_cf),
          dim=1)
      if self.args["with_sin"]:
        image_features = self.expand_sin(image_features)

    mid_outputs = {
      "cost_volume": cost_volume,
      "image_features_1": image_features,
    }
    #mono-stereo fuse layer:
    for i in range(len(self.decoders2)):
      image_features, _ = self.decoders2[i](image_features)
    
    # mid_outputs["image_features_2"] = image_features

    assert torch.isfinite(image_features).all(), "Nan in image features"
    # Here the cost volume already takes into account the translation.    
    if self.args["mvs_uncertainty"]:
      # print("image_features[:, :1].shape:", image_features[:, :1].shape)
      pred = self.activation_G(image_features) #out_type: depth
      pred_d = pred[:, :1, ...]
      if self.args["out_type"] == "disparity":
        eps=1e-10
        rectified_depth = 1.0/ (torch.clamp(pred_d, min=0)+eps)
      elif self.args["out_type"] == "depth":
        rectified_depth = torch.clamp(pred_d, min=0)
      pred_final = torch.cat([rectified_depth, pred[:, 1:, ...]], dim=1)
    else:
      if self.args["out_type"] == "disparity":
        eps=1e-10
        rectified_depth = 1.0/ (torch.clamp(image_features[:, :1], min=0)+eps)
      elif self.args["out_type"] == "depth":
        rectified_depth = torch.clamp(image_features[:, :1], min=0)

    rectified_depth = rectified_depth.permute((0, 2, 3, 1))
    depth = rectified_depth
    cost_reg = cost_reg.permute((0, 2, 3, 1))
    return_data = {
      "depth": depth,
      "rectified_depth_d1": rectified_depth_d1,
      "rectified_depth": rectified_depth,
      "raw_image_features": image_features.permute((0, 2, 3, 1)),
      "raw_image_features_d1": raw_image_features_d1,
      "cost_reg": cost_reg,
      "mid_outputs": mid_outputs
    }
    if self.args["mvs_uncertainty"]:
      return_data["pred_final"] = pred_final
    if self.args["contain_dnet"]:
      ref_mu = ref_mu.permute((0, 2, 3, 1))
      ref_mu = ref_mu.permute((0, 3, 1, 2))
      return_data["mono_depth_ref"] = ref_mu
    return return_data





  def activation_G(self, out):
    mu, var = torch.split(out, 1, dim=1)
    var = F.elu(var) + 1.0 + 1e-10
    out = torch.cat([mu, var], dim=1)  # (N, 2, H, W)
    return out


  def initialize_omega(self, dtype, height, width):
    if self.omega is None:
      omega = torch.arange(0, height, device=self.device, dtype=dtype) + 0.5
      omega = omega * np.pi / height
      omega = omega.view(1, 1, height, 1).repeat(1, 1, 1, width)
      # print("Omega shape", omega.shape)
      self.omega = omega
    return self.omega

  def calculate_cost_volume(self,
                            images,
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
    grid_x = torch.linspace(-1, 1, width, device=self.device)
    grix_y = torch.linspace(-1, 1, height, device=self.device)
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

  def calculate_cost_volume_erp(self,
                                images,
                                depths,
                                trans_norm,
                                cost_type="abs_diff",
                                direction="up"):
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

  def backwards_warping_uv(self, target_depth, rot, trans, inv_rot=True):
    """Does backwards warping returning the UV values.

    Args:
      target_depth: target depth. Should be (B, H, W) tensor.
      rot: rotation. Should be a (B, 3, 3) tensor.
      trans: translation. Should be (B, 3) tensor.
      inv_rot: invert the rotation.

    Returns:
      UV coordinates of the source image in range (0, 2*pi) x (0, pi)

    """
    batch_size, height, width = target_depth.shape[:3]

    phi = torch.arange(0,
                       height,
                       device=target_depth.device,
                       dtype=target_depth.dtype)
    phi = (phi + 0.5) * (np.pi / height)
    theta = torch.arange(0,
                         width,
                         device=target_depth.device,
                         dtype=target_depth.dtype)
    theta = (theta + 0.5) * (2 * np.pi / width) + np.pi / 2
    phi, theta = torch.meshgrid(phi, theta)

    xyz = my_torch_helpers.spherical_to_cartesian(theta[None, :, :],
                                                  phi[None, :, :], target_depth)

    if inv_rot:
      # rot_inv = torch.inverse(rot)
      rot_inv = torch.linalg.inv(rot)
      

      xyz = torch.matmul(rot_inv[:, None, None, :, :], xyz[:, :, :, :,
                                                       None])[:, :, :, :, 0]
      xyz = xyz + trans[:, None, None, :]
    else:
      xyz = xyz + trans[:, None, None, :]
      xyz = torch.matmul(rot[:, None, None, :, :], xyz[:, :, :, :,
                                                   None])[:, :, :, :, 0]

    uvs = my_torch_helpers.cartesian_to_spherical(xyz)
    u = torch.fmod(uvs[:, :, :, 0] - np.pi / 2 + 4 * np.pi, 2 * np.pi)
    v = uvs[:, :, :, 1]
    uvs = torch.stack((u, v), dim=-1)
    return uvs

  def backwards_warping(self, images, target_depth, rot, trans, inv_rot=True):
    """Does backwards warping.

    Args:
      images: Panoramas
      target_depth: Target depth. Should be BxHxW.
      rot: Rotation. Should be Bx3x3.
      trans: Translation. Should be Bx3.
      inv_rot: Invert the rotation.

    Returns:
      RGB images as BxHxWxC tensor.

    """
    full_height, full_width = images.shape[1:3]
    batch_size, height, width = target_depth.shape[:3]
    uvs = self.backwards_warping_uv(target_depth, rot, trans, inv_rot=inv_rot)
    uvs = uvs * torch.tensor([full_width / (2 * np.pi), full_height / np.pi],
                             dtype=images.dtype,
                             device=images.device)
    uvs = uvs - 0.5
    uvs = uvs * torch.tensor([2 / full_width, 2 / full_height],
                             dtype=images.dtype,
                             device=images.device) - 1.0
    images = F.grid_sample(images.permute((0, 3, 1, 2)),
                           uvs,
                           mode="bilinear",
                           align_corners=False)
    return images.permute(0, 2, 3, 1)

  def render_point_cloud(self, pointcloud, rot=None, trans=None,
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
    # print("Rots shape", rot.shape)
    # print("Trans shape", trans.shape)
    cameras = OpenGLPerspectiveCameras(device=self.device,
                                       R=rot,
                                       T=trans,
                                       fov=99999,
                                       znear=0.0000001,
                                       zfar=100.0)
    raster_settings = PointsRasterizationSettings(
      image_size=self.raster_resolution,
      radius=self.point_radius,
      points_per_pixel=self.points_per_pixel)
    rasterizer = SpherePointsRasterizer(cameras=cameras,
                                        raster_settings=raster_settings,
                                        linearize_angle=linearize_angle)
    renderer = SpherePointsRenderer(rasterizer=rasterizer,
                                    compositor=AlphaCompositor())
    return renderer(pointcloud)


  def forward(self, x_in):
    """Performs a forward pass and return the rendered point cloud.

    Args:
      x_in: Panos, rotations, translations as a tuple.

    Returns:
      Rendered point cloud.

    """
    panos, rots, trans = x_in

    batch_size, seq_len, height, width = panos.shape[:4]
    # print("batch size", batch_size, seq_len, height, width)
    depths = self.estimate_depth(
      panos.reshape((batch_size * seq_len, height, width, 3)))
    depths = depths.reshape((batch_size, seq_len, height, width))
    if self.verbose:
      print("Average depths", np.mean(depths.detach().cpu().numpy()))
    verts, colors = self.make_point_cloud(depths, panos, rots, 0.5 * trans)

    images = self.render_point_cloud(verts, colors)
    return images

  def predict_with_backwards_warping(self, panos, rots, trans, full_panos):
    """Performs prediction with backwards warping.

    Args:
      panos: Input panos, shrunken
      rots: Input rotations
      trans: Input translations
      full_panos: Full resolution panos

    Returns:
      Warped images

    """
    height = self.raster_resolution
    width = self.raster_resolution

    batch_size, seq_len, full_height, full_width = panos.shape[:4]

    depths = self.estimate_depth(panos[:, [0, 2], :, :, :].reshape(
      (batch_size * 2, height, width, 3)))
    depths = depths.reshape((batch_size, 2, height, width))
    outputs1 = self.backwards_warping(full_panos[:, 1, :, :, :],
                                      depths[:, 0, :, :], rots[:, 0, :, :],
                                      0.5 * trans[:, 0, :])
    outputs2 = self.backwards_warping(full_panos[:, 1, :, :, :],
                                      depths[:, 1, :, :], rots[:, 2, :, :],
                                      0.5 * trans[:, 2, :])
    return torch.stack((outputs1, outputs2), dim=1)

  def do_inpainting(self, input):
    """Runs the inpainting UNet.

    Args:
      input: input image. Channels last.

    Returns:
      Output image. Channels last.

    """
    return self.inpaint_net(input)

  def estimate_pose(self, images):
    """Estimates the pose using Monodepth2's pose estimator

    Args:
      images: Set of 2 images. Bx2xHxWxC.

    Returns:
      rotations: Bx3x3
      translation: Bx3

    """

    m_images = torch.cat((images[:, 0, :, :, :], images[:, 1, :, :, :]), dim=3)
    m_images = m_images.permute((0, 3, 1, 2))
    shrinked_input_images = torch.nn.functional.interpolate(m_images,
                                                            (320, 1024),
                                                            mode="bilinear",
                                                            align_corners=False)

    features = [self.pose_encoder(shrinked_input_images)]
    axisangle, translation = self.pose_decoder(features)

    rotation = rot_from_axisangle(axisangle[:, 0])
    rotation = rotation[:, :3, :3]

    translation = 25 * translation[:, 0, 0]

    return rotation, translation

  def correct_depths(self, depths, panos1, panos0, rot, trans, far_depth=100):
    """This sets the depths to infinity if the backward warping loss is smaller for the unwarped image.

    Args:
      depths: Depths of pano1. Should be BxHxWx1.
      panos1: Pano1. BxHxWxC.
      panos0: Pano0. BxHxWxC
      rot: Rotation from pano0 to pano1. Should be (B, 3, 3) tensor.
      trans: Translation from pano0 to pano1. Should be (B, 3) tensor.

    Returns:
      Depths of pano1 as (B, H, W, C) tensor.

    """
    print("Shapes", depths.shape, panos1.shape, panos0.shape, rot.shape,
          trans.shape)
    y_pred = self.backwards_warping(panos0, depths[:, :, :, 0], rot, trans)
    y_true = panos1
    y_pred_org = panos0
    loss_pred = torch.mean(torch.abs(y_pred - y_true), dim=3)
    loss_orig = torch.mean(torch.abs(y_pred_org - y_true), dim=3)
    far_depth_tensor = torch.tensor(far_depth,
                                    dtype=depths.dtype,
                                    device=depths.device).expand_as(depths)
    far_indices = torch.lt(loss_orig, loss_pred)[:, :, :, None]
    print("Shapes", far_indices.shape, depths.shape)
    new_depths = torch.where(far_indices, far_depth_tensor, depths)
    return new_depths

  def build_depth_mask(self,
                       depth_img,
                       threshold=1,
                       use_second_depth=True,
                       dtype=torch.float32):
    """Builds a depth mask from the image.

    Args:
      depth_img: Depth image as (N, H, W, K) tensor.
      threshold: Threshold.
      use_second_depth: Whether to consider the depth of the second point.
      dtype: Return dtype.

    Returns: Depth mask as (N, H, W, 1) tensor.

    """
    mask1 = torch.lt(depth_img[:, :, :, 0:1], threshold)
    final_mask = mask1
    if use_second_depth:
      mask2 = torch.lt(depth_img[:, :, :, 1:2], threshold)
      final_mask = mask1 | mask2
    return final_mask.type(dtype)

  # def rectify_images(self, panos, rots, trans, include_rots=False,
  #                    rectify_vertical=True):
  #   """Rectifies images. Second pano should be a reference pano.

  #   Args:
  #     panos: Panos as (B, 2, H, W, C) tensor.
  #     rots: Rotations as (B, 2, 3, 3) tensor.
  #     trans: Translation as (B, 2, 3, 3) tensor.
  #     include_rots: Debugging flag.
  #     rectify_vertical: Rectify upwards.

  #   Returns:
  #     Rectified panos as (B, 2, H, W, C) tensor.
  #     Rotation matrices as (B, 2, 3, 3) tensor.

  #   """
  #   batch_size, _, height, width, channels = panos.shape
  #   # Calculate rotation matrices so that the movement is forward.
  #   rot_fw = []
  #   for i in range(batch_size):
  #     m_trans = (trans[i, 0] - trans[i, 1]).cpu().numpy()
  #     m_trans = m_trans / np.linalg.norm(m_trans)
      
  #     # if i==1:
      
  #     m_ang = np.arccos(
  #       -m_trans[2] / np.linalg.norm(m_trans[[0, 2]])) * (2 * (-m_trans[0] > 0) - 1)
      
  #     # print("Angle", m_trans, np.rad2deg(m_ang))
  #     m_rot_fw = my_helpers.rotate_around_axis(np.array([0, 1, 0]), m_ang)
  #     rot_fw.append(
  #       torch.tensor(m_rot_fw, device=panos.device, dtype=torch.float32))

  #   rot_fw = torch.stack(rot_fw, dim=0)
  #   # Calculate rotation matrices for images 0 and 1.
  #   if rectify_vertical:
  #     # Rotation matrix to make forward face down.
  #     rot_up = my_helpers.rotate_around_axis(np.array([1, 0, 0]), np.pi / 2)
  #     rot_up = torch.tensor(rot_up, device=panos.device, dtype=torch.float32)
  #     rot_up = rot_up.view(1, 3, 3).repeat(batch_size, 1, 1) 
  #     rot_0 = rots[:, 0] @ rot_fw @ rot_up
  #     rot_1 = rots[:, 1] @ rot_fw @ rot_up
  #   else:
  #     rot_0 = rots[:, 0] @ rot_fw
  #     rot_1 = rots[:, 1] @ rot_fw

  #   rect_panos_0 = my_torch_helpers.rotate_equirectangular_image(
  #     panos[:, 0], rot_0.float(), linearize_angle=np.deg2rad(5))
  #   rect_panos_1 = my_torch_helpers.rotate_equirectangular_image(
  #     panos[:, 1], rot_1.float(), linearize_angle=np.deg2rad(5))
  #   rect_panos = torch.stack((rect_panos_0, rect_panos_1), dim=1)
  #   rect_rots = torch.stack((rot_0, rot_1), dim=1)
    
  #   if include_rots:
  #     return rect_panos, rect_rots, rot_up, rot_fw
  #   return rect_panos, rect_rots

  # def unrectify_image(self, pano, rot_mat):
  #   """Undo a rectification.

  #   Args:
  #     pano: Panos to derotate as (B, H, W, C) tensor.
  #     rot_mat: Rotation matrix for rectification as (B, 3, 3) tensor.

  #   Returns:
  #     Unrotated pano as (B, H, W, C) tensor.
  #   """
  #   # rot_inv = torch.inverse(rot_mat)# not robust for the precision
  #   try:
  #     # print("rot_mat:", rot_mat)
  #     # rot_inv = torch.linalg.pinv(rot_mat.double()).float()
  #     # rot_inv = torch.inverse(rot_mat.double()).float()# not robust for the precision

  #     rot_inv = torch.inverse(rot_mat)# not robust for the precision

  #     # print("rot_inv:", rot_inv)
  #   except Exception:
      
  #     print("inverse Error")

  #   unrot_pano = my_torch_helpers.rotate_equirectangular_image(pano, rot_inv)
  #   return unrot_pano

  def get_total_params(self):
    """Gets the total number of parameters.

    Returns:
      int total number of parameters.
    """
    unet_params = my_torch_helpers.total_params(self.unet)
    unet3d_params = my_torch_helpers.total_params(self.unet3d)
    decoders1_params = my_torch_helpers.total_params(self.decoders1)
    decoders2_params = my_torch_helpers.total_params(self.decoders2)
    total_params = unet_params + unet3d_params + decoders1_params + decoders2_params
    # if self.args[""]
    args = self.args
    if args["contain_dnet"]:
      total_params += my_torch_helpers.total_params(self.d_net)
    return total_params



  def forward(self, panos, rots, trans, min_depth=2.0, max_depth=100):

    """Estimates depth of images[:, 1] using a cost volume.

    Args:
      panos: Tensor of shape (B, 2, H, W, 3)
      rots: Tensor of shape (B, 2, 3, 3)
      trans: Tensor of shape (B, 2, 3)

    Returns:
      Estimated depth.

    """

    # panos = panos.permute((0, 1, 3, 4, 2))
    args = self.args


    batch_size, seq_len, height, width, _ = panos.shape
    # print("height, width:", height, width)
    B = batch_size

    # trans_norm = torch.norm(trans[:, 0] - trans[:, 1], dim=1)
    
    # rectified_panos, rect_rots = self.rectify_images(panos, rots, trans,
    #                                                  rectify_vertical=False)

    
    # Change to channels first.
    panos_cf = panos.permute((0, 1, 4, 2, 3))
    if self.args["contain_dnet"]:
      #mono-net
      args = self.args
      mono_panos_cf = panos_cf.reshape(batch_size*seq_len, 3, args["height"], args["width"])
      mono_panos_cf = torch.nn.functional.interpolate(mono_panos_cf,
                                                (args["mono_height"], args["mono_width"]),
                                                mode=args["interpolation_mode"],
                                                align_corners=False)
      mono_panos_cf = mono_panos_cf.reshape(batch_size, seq_len, 3, args["mono_height"], args["mono_width"])
      mono_panos_cf = mono_panos_cf.permute((0, 1, 3, 4, 2))
      mono_panos_cube = self.e2c_process(mono_panos_cf, self.e2c_mono)


    if self.args["contain_dnet"]:
      with torch.no_grad():
        # D-Net forward pass
        # Change to channels first.
        mono_panos_cf = mono_panos_cf.permute((0, 1, 4, 2, 3))
        mono_panos_cube = mono_panos_cube.permute((0, 1, 4, 2, 3))
        mono_erp_inputs, mono_cube_inputs = self.normalize_input(mono_panos_cf, mono_panos_cube)
        mono_erp_inputs = torch.cat((mono_erp_inputs[:, 1], mono_erp_inputs[:, 0]), dim=0)
        mono_cube_inputs = torch.cat((mono_cube_inputs[:, 1], mono_cube_inputs[:, 0]), dim=0)
        mono_outputs = self.d_net(mono_erp_inputs, mono_cube_inputs, dnet=True)
        x_d3 = mono_outputs["mono_feat"]
        if self.args["mono_uncertainty"]:
          mono_gmms = mono_outputs["pred"]
        else:
          mono_gmms = mono_outputs["pred_depth"]
        mono_gmms_orig = mono_gmms.clone()
        mono_gmms = F.interpolate(mono_gmms, [height//4, width//4], mode='bilinear')
        x_d3 = F.interpolate(x_d3, [height//4, width//4], mode='bilinear')
        if self.args["mono_uncert_tune"]:
          mono_std = mono_outputs["mono_std"]
          mono_std = F.interpolate(mono_std, [height//4, width//4], mode='bilinear')
    
        mono_gmms = mono_gmms.detach()
        ref_gmms = mono_gmms[:B, ...]
        x_d3 = x_d3[:B, ...]
        nghbr_gmms = mono_gmms[B:, ...]
        if self.args["mono_uncert_tune"]:
          ref_sigma = mono_std[:B, ...]

    #   # pred_list = [ref_gmms]
      ref_mu = ref_gmms[:, :1, ...]
      if self.args["mono_uncertainty"]:
        ref_sigma = ref_gmms[:, 1:, ...]

    #   #todo
      # ref_sigma = (max_depth - min_depth)/self.n_samples
      if self.n_samples>0:
        if self.args["mono_uncertainty"] or args["mono_uncert_tune"]:
          if args["relaxation_factor"] in self.args:
            print("use relaxation_factor:", args["relaxation_factor"])
            
            depth_volume = [torch.clamp(ref_mu + ref_sigma * k * self.args["relaxation_factor"], min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
          else:
            depth_volume = [torch.clamp(ref_mu + torch.clamp(ref_sigma, min=args["basic_sigma"]) * k, min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
        else:
          depth_volume = [torch.clamp(ref_mu + k*self.args["fixed_sigma"], min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
        depth_volume = torch.cat(depth_volume, dim=1)                       # B, N_samples, H, W    



    image_features_0_cf = panos_cf[:, 0]
    image_features_1_cf = panos_cf[:, 1]
    # Pass through encoding UNet.
    if self.args["net"]=="OmniSyn":
      image_features_0_cf = self.unet(image_features_0_cf)
      image_features_1_cf = self.unet(image_features_1_cf)
    else:
      panos_cube = self.e2c_process(panos, self.e2c)    
      panos_cube = panos_cube.permute((0, 1, 4, 2, 3))
      image_features_0_cube=panos_cube[:, 0]
      image_features_1_cube=panos_cube[:, 1]
      if self.args["contain_dnet"] and self.args["change_input"]:
        ref_mu_orig = mono_gmms_orig[:B, :1, ...]
        other_mu_orig = mono_gmms_orig[B:, :1, ...]

        # depth_image = torch.cat([])
        depth_image_1 = input_normalize_depth(args, ref_mu_orig)
        depth_image_0 = input_normalize_depth(args, other_mu_orig)
        image_features_0_cf = self.unet(image_features_0_cf, image_features_0_cube, depth_image_0)
        image_features_1_cf = self.unet(image_features_1_cf, image_features_1_cube, depth_image_1)

      else:
        image_features_0_cf = self.unet(image_features_0_cf, image_features_0_cube)
        image_features_1_cf = self.unet(image_features_1_cf, image_features_1_cube)

    # Change to channels last.
    image_features_0 = image_features_0_cf.permute((0, 2, 3, 1))#B, H, W, 1
    image_features_1 = image_features_1_cf.permute((0, 2, 3, 1))#B, H, W, 1

    # Create cost volume.
    image_features = torch.stack((image_features_0, image_features_1), dim=1)
    
    # import pdb;pdb.set_trace()
    # cost_volume = self.calculate_cost_volume_erp(
    #   image_features,
    #   depths=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, 64),
    #   trans_norm=trans_norm,
    #   cost_type='abs_diff')
    if self.contain_dnet:
      thres = int(self.weighting.split('CW')[1])
      if args["wo_hdh"]:
        d_centers = None
        pass
        # depth_volume = torch.cat([depth_volume, d_centers], dim=1)      
      else:
        n_depths = self.cost_volume_channels - self.n_samples
        if args["use_depth_sampling"]:
          if args["revise_range"]:
            # [-fixed_sigma, + fixed_sigma]
            # d_centers = ref_mu + fixed_sigma * k
            fixed_dist = args["fixed_dist"]
            #B, 1, H, W
            d_min = torch.clamp(ref_mu - fixed_dist, min=args["min_depth"])
            d_max = torch.clamp(ref_mu + fixed_dist, max=args["max_depth"])
            # d_centers = [torch.clamp(ref_mu + ref_sigma * k * self.args["relaxation_factor"], min=args["min_depth"], max=self.args["max_depth"]) for k in self.k_list]
            # revise_min_depth =             
            # d_centers=torch.linspace(min_depth, max_depth, n_depths).reshape(1, n_depths, 1, 1).cuda()
            interval = (d_max-d_min)/(n_depths-1) # B, 1, H, W
            # start = d_min
            vals = torch.arange(0, n_depths-1).reshape(1, n_depths-1, 1, 1).cuda()#1, D-1, H, W
            adds = interval * vals #B, D-1, H, W            
            d_centers = torch.cat([d_min, d_min+adds], dim=1)
          else:
            d_centers=torch.linspace(min_depth, max_depth, n_depths).reshape(1, n_depths, 1, 1).cuda()          
        else:
          d_centers=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, n_depths).reshape(1, n_depths, 1, 1).cuda()
        
        # B, D, H, W = depth_volume.shape
        B, _, H, W = ref_mu.shape
        if args["revise_range"]:
          pass
        else:
          d_centers = d_centers.repeat(B, 1, H, W)                
        
        if self.n_samples>0:
          depth_volume = torch.cat([depth_volume, d_centers], dim=1)
        else:
          depth_volume = d_centers #torch.cat([depth_volume, d_centers], dim=1)
        depth_volume, _ = torch.sort(depth_volume, dim=1)#important!!! avoid overfit to mono depth hypotheses
    else:
      n_depths = self.cost_volume_channels
      if args["use_depth_sampling"]:
        d_centers=torch.linspace(min_depth, max_depth, n_depths)#.reshape(1, n_depths, 1, 1).cuda()
      else:
        d_centers=1.0 / torch.linspace(1 / min_depth, 1 / max_depth, n_depths)#.reshape(1, n_depths, 1, 1).cuda()
    if self.contain_dnet:
      cost_volume = calculate_cost_volume_erp(self.args,
        image_features,
        depths=d_centers,
        depth_volume=depth_volume,
        trans = trans,
        rots = rots,
        cost_type='abs_diff',
        ref_gmms=ref_gmms,
        nghbr_gmms = nghbr_gmms,
        thres=thres)
    else:
      cost_volume = calculate_cost_volume_erp(self.args,
        image_features,
        depths=d_centers,
        trans = trans,
        rots = rots,
        cost_type='abs_diff')
    # Pass cost volume through Conv3D layers.
    cost_volume = cost_volume.permute((0, 4, 1, 2, 3))
    cost_volume = self.unet3d(cost_volume)
    # Now the length dimension is now the channel dimension.
    image_features = cost_volume[:, 0, :, :, :]
    cost_reg = image_features #B, D, H, W
    # if args["mvs_uncert"]:

    #   raw_image_features_d1 = self.decoders1(torch.cat(
    #         (cost_volume[:, 0, :, :, :], image_features_1_cf),
    #         dim=1))
  # else: 
    raw_image_features_d1 = self.decoders1(image_features)
    raw_image_features_d1 = nn.functional.interpolate(
      raw_image_features_d1,
      scale_factor=4,
      mode="bilinear",
      align_corners=False
    ).permute((0, 2, 3, 1))

    # Here the cost volume already takes into account the translation.
    if self.args["out_type"]=='disparity':
      eps=1e-10
      rectified_depth_d1 = 1.0 / (torch.clamp(raw_image_features_d1, min=0)+eps)
    elif self.args["out_type"] == 'depth':
      rectified_depth_d1 = torch.clamp(raw_image_features_d1, min=0)

    # Now the length dimension is now the channel dimension.
    # image_features = cost_volume[:, 0, :, :, :]
    if self.args["wo_mono_feat"]:
      image_features = cost_volume[:, 0, ...]

    else:
      if self.args["contain_dnet"]:
        image_features = torch.cat(
          (cost_volume[:, 0, :, :, :], x_d3),
          dim=1)
      else:  
        image_features = torch.cat(
          (cost_volume[:, 0, :, :, :], image_features_1_cf),
          dim=1)
      if self.args["with_sin"]:
        image_features = self.expand_sin(image_features)

    mid_outputs = {
      "cost_volume": cost_volume,
      "image_features_1": image_features,
    }
    #mono-stereo fuse layer:
    for i in range(len(self.decoders2)):
      image_features, _ = self.decoders2[i](image_features)
    
    # mid_outputs["image_features_2"] = image_features

    assert torch.isfinite(image_features).all(), "Nan in image features"
    # Here the cost volume already takes into account the translation.    
    if self.args["mvs_uncertainty"]:
      # print("image_features[:, :1].shape:", image_features[:, :1].shape)
      pred = self.activation_G(image_features) #out_type: depth
      pred_d = pred[:, :1, ...]
      if self.args["out_type"] == "disparity":
        eps=1e-10
        rectified_depth = 1.0/ (torch.clamp(pred_d, min=0)+eps)
      elif self.args["out_type"] == "depth":
        rectified_depth = torch.clamp(pred_d, min=0)
      pred_final = torch.cat([rectified_depth, pred[:, 1:, ...]], dim=1)
    else:
      if self.args["out_type"] == "disparity":
        eps=1e-10
        rectified_depth = 1.0/ (torch.clamp(image_features[:, :1], min=0)+eps)
      elif self.args["out_type"] == "depth":
        rectified_depth = torch.clamp(image_features[:, :1], min=0)

    rectified_depth = rectified_depth.permute((0, 2, 3, 1))
    depth = rectified_depth
    cost_reg = cost_reg.permute((0, 2, 3, 1))
    return_data = {
      "depth": depth,
      "rectified_depth_d1": rectified_depth_d1,
      "rectified_depth": rectified_depth,
      "raw_image_features": image_features.permute((0, 2, 3, 1)),
      "raw_image_features_d1": raw_image_features_d1,
      "cost_reg": cost_reg,
      "mid_outputs": mid_outputs
    }
    if self.args["mvs_uncertainty"]:
      return_data["pred_final"] = pred_final
    if self.args["contain_dnet"]:
      ref_mu = ref_mu.permute((0, 2, 3, 1))
      ref_mu = ref_mu.permute((0, 3, 1, 2))
      return_data["mono_depth_ref"] = ref_mu
    return return_data

