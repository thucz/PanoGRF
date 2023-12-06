import torch.nn as nn
from .my_equi import Equi
from .my_unifuse import UniFuse
from .my_erp_tp import ERP_TP_Fuse
from .my_tp_only import TP
from .my_cube_only import Cube
from models.common_blocks import (ConvBlock, Conv3DBlock, Conv3DBlockv2,
                                  ConvBlock2, UNet2)
from .cost_reg import CostRegNet
                                  
from sync_batchnorm import convert_model

def init_encoders(args):
    # network
    Net_dict = {"UniFuse": UniFuse,
                "Equi": Equi,
                "ERP+TP":ERP_TP_Fuse,
                "TP":TP,
                "Cube":Cube}

    Net = Net_dict[args["net"]]
    if args["net"]=="UniFuse":
        model = Net(args["num_layers"], args["height"], args["width"],
                            args["imagenet_pretrained"],
                            fusion_type=args["fusion"], se_in_fusion=args["se_in_fusion"], use_wrap_padding=True)
    elif args["net"]=="Cube":
        model = Net(args["num_layers"], args["height"], args["width"],
                            args["imagenet_pretrained"],
                            fusion_type=args["fusion"], se_in_fusion=args["se_in_fusion"], use_wrap_padding=True)
    
    elif args["net"]=="ERP+TP":
        fov = (args["fov"], args["fov"])#(48, 48)
        patch_size = args["patchsize"]
        nrows = args["nrows"]
        npatches_dict = {3:10, 4:18, 5:26, 6:46}
        # num_gpu = torch.cuda.device_count()
        network = ERP_TP_Fuse(args["num_layers"], args["height"], args["width"],
                            args["imagenet_pretrained"],
                            fusion_type=args["fusion"], se_in_fusion=args["se_in_fusion"], use_wrap_padding=True,
                            nrows=nrows, npatches=npatches_dict[nrows], patch_size=patch_size, fov=fov)
        model = convert_model(network)
    elif args["net"]=="TP":
        fov = (args["fov"], args["fov"])#(48, 48)
        patch_size = args["patchsize"]
        nrows = args["nrows"]
        npatches_dict = {3:10, 4:18, 5:26, 6:46}
        # num_gpu = torch.cuda.device_count()
        network = TP(args["num_layers"], args["height"], args["width"],
                            args["imagenet_pretrained"],
                            fusion_type=args["fusion"], se_in_fusion=args["se_in_fusion"], use_wrap_padding=True,
                            nrows=nrows, npatches=npatches_dict[nrows], patch_size=patch_size, fov=fov)
        model = convert_model(network)

    elif args["net"]=="Equi":
        model = Net(args["num_layers"], args["height"], args["width"],
                        args["imagenet_pretrained"], use_wrap_padding=True, with_sin=args["with_sin"])
    model.to(args["device"])

    return model

def initialize_cost_volume_network(args, layers=5, size=4,
                                        use_wrap_padding=True,
                                        use_v_input=False,
                                        out_channels=1, cost_volume_channels=64, input_option='erp+cube'):
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
    encoder = init_encoders(args)    
    
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
    if "use_new_reg3dnet" in args and args["use_new_reg3dnet"]: #todo: use_new_reg3dnet, 
      if args["group_wise"]:
        in_channels = 64//args["group_num"]
      else:
        in_channels = 64
      unet3d = CostRegNet(args, in_channels=in_channels)
    else:
      if "cnn3d_num_layer" in args:
        cnn3d_num_layer = args["cnn3d_num_layer"]
      else:
        cnn3d_num_layer = 3

      for i in range(0, cnn3d_num_layer):
        # if i==0:
        #   channels = cost_volume_channels
        # else:
        channels = 2 ** (i + size + 1)
        cv_encoders.append(
          Conv3DBlockv2(in_channels=channels, #2**(0+4+1) = 32
                        out_channels=2 * channels, #
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
        Conv3DBlockv2(in_channels=2 ** (cnn3d_num_layer + size + 1),
                      out_channels=2 ** (cnn3d_num_layer + size + 2),
                      kernel_size=(3, 3, 3),
                      stride=(1, 1, 1),
                      padding=(1, 1, 1),
                      use_batch_norm=False,
                      pooling=nn.Identity(),
                      use_wrap_padding=use_wrap_padding,
                      use_v_input=use_v_input))
      unet3d = UNet2(nn.ModuleList(cv_encoders),
                    nn.ModuleList(cv_decoders),
                    interpolation="trilinear",
                    name="unet3d")

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

    if args["wo_mono_feat"]:
      in_dim = cost_volume_channels
    else:
      in_dim = cost_volume_channels + 2 ** (size + 1) #cost_volume_channels
    if args["with_sin"]:
      in_dim += 1
    decoders2 = [
      ConvBlock2(in_channels=in_dim,
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

    
    decoders2 = nn.ModuleList(decoders2)
    return encoder, unet3d, decoders1, decoders2

def initialize_unet(args, layers=5, size=4,
                                        use_wrap_padding=True,
                                        use_v_input=False,
                                        out_channels=1, cost_volume_channels=64, input_option='erp+cube'):
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
    encoder = init_encoders(args)    
    
    return encoder
