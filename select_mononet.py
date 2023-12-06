import torch
import torch.nn as nn
import os
def select_mono(args, mvsnet=False):
    if args["mono_net"] == "UniFuse" or args["mono_net"] == "Equi":
        import sys
        sys.path.append("./UniFuse-Unidirectional-Fusion/UniFuse")
        # from datasets.util import Equirec2Cube
        from networks import UniFuse, Equi
        from networks.convert_module import erp_convert
        from networks.layers import Conv3x3, Conv3x3_wrap

        Net_dict = {"UniFuse": UniFuse,
                "Equi": Equi}
        #todo: args["net"],num_layers, imagenet_pretrained, se_in_fusion, fusion
        Net = Net_dict[args["mono_net"]]
        if mvsnet:
            model = Net(args["mono_num_layers"], args["mono_height"], args["mono_width"],
                            args["imagenet_pretrained"], args["max_depth"],
                            fusion_type=args["mono_fusion"], se_in_fusion=args["se_in_fusion"], mono_uncertainty=args["mono_uncertainty"], mono_lowres_pred=False)
        else:
            model = Net(args["num_layers"], args["mono_height"], args["mono_width"],
                            args["imagenet_pretrained"], args["max_depth"],
                            fusion_type=args["fusion"], se_in_fusion=args["se_in_fusion"], mono_uncertainty=args["mono_uncertainty"], mono_lowres_pred=False)

        if args["load_from_pretrained"]:
            #use pretrained mono model
            from load_dnet_model import load_model
            model = load_model(model, args["load_weights_dir"])#todo
            # use wrap padding

        if args["use_wrap_padding"]:
            model.equi_encoder = erp_convert(model.equi_encoder)
            model.equi_decoder = erp_convert(model.equi_decoder)
            
        if args["mono_uncertainty"]:
            # model.uncertainty_conv = Conv3x3(16, 1).cuda()
            model.equi_dec_convs["depthconv_0"] = Conv3x3_wrap(16, 2).cuda()

        if args["mono_uncert_tune"]:
            # load_mvs_model(self.mvs_net, args["mvs_checkpoints_dir"])
            
            # for param in self.mvs_net.parameters():
            #   param.requires_grad = False
            # self.mvs_net.eval()
            from network.omni_mvsnet.mono_uncert_wrapper import MonoUncertWrapper
            model = MonoUncertWrapper(args, model)

        # if args["mono_lowres_pred"]:
        #     model.lowres_depth_conv = ConvBlock(
        #         32,
        #         2,
        #         kernel_size=1,
        #         padding=0,
        #         stride=1,
        #         upscale=False,
        #         gate=False,
        #         use_wrap_padding=False,
        #         use_batch_norm=False,
        #         use_activation=False,
        #     ).cuda()
            
    elif args["mono_net"] == "PanoFormer":
        import sys
        sys.path.append("./PanoFormer/PanoFormer")
        from network.model import Panoformer as PanoBiT
        model = PanoBiT()
        if args["load_from_pretrained"]:
            #use pretrained mono model
            from load_dnet_model import load_model
            model = load_model(model, args["load_weights_dir"])#todo
    elif args["mono_net"] == "FreDSNet":
        import sys
        sys.path.append("./FreDSNet/")
        import FreDSNet_model as fre_model
        model,state_dict = fre_model.load_weigths(args)
    elif args["mono_net"] == "ACDNet":
        import sys
        sys.path.append("./ACDNet/")
        # import ipdb;ipdb.set_trace()
        # import sys
        # _cpath_ = sys.path[0] #获取当前路径
        # sys.path.remove(_cpath_) #删除
        from acd_models.acdnet.acdnet import ACDNet
        # from jira import JIRA
        # sys.path.insert(0, _cpath_) #恢复

        model = ACDNet()
        state_dict = torch.load(args['checkpoints'],map_location='cpu')
        model.load_state_dict(state_dict['model'],strict=True)
    elif args["mono_net"] == "Joint":
        import sys
        sys.path.append("./Joint_360depth")
        from DPT.dpt.models import DPTDepthModel
        model = DPTDepthModel(
                    path=None,
                    backbone="vitb_rn50_384",
                    non_negative=True,
                    enable_attention_hooks=False,
                        )
        model.load_state_dict(torch.load(args["mono_checkpoints"]))
    elif args["mono_net"] == "HRDFuse":
        import sys
        sys.path.append("./HRDFuse_github")
        from sync_batchnorm import convert_model
        from model.spherical_fusion import spherical_fusion
        fov = (args["fov"], args["fov"])  # (48, 48)
        patch_size = args["patch_size"]#(args["patchsize"], args["patchsize"])
        nrows = args["nrows"] #
        npatches_dict = {3: 10, 4: 18, 5: 26, 6: 46} #
        iters = args["iter"] #
        min_val = args["min_val"] # 0.1 
        max_val = args["max_val"] # 10

        network = spherical_fusion(nrows=nrows, npatches=npatches_dict[nrows], patch_size=patch_size, fov=fov, min_val=min_val,
                           max_val=max_val)
        network = convert_model(network)
        network = nn.DataParallel(network)
        # network.cuda()
        if args["mono_checkpoint"] is not None:
            print("loading model from folder {}".format(args["mono_checkpoint"]))
            if os.path.isfile(args["mono_checkpoint"]):
                path = args["mono_checkpoint"]
            else:
                path = os.path.join(args["mono_checkpoint"], "{}.tar".format("checkpoint_best"))
            
            model_dict = network.state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            network.load_state_dict(model_dict)
            model = network

    # model = Net(args["mono_num_layers"], args["height"], args["width"],
    #               args["imagenet_pretrained"], args["max_depth"],
    #               fusion_type=args["fusion"], se_in_fusion=args["se_in_fusion"], use_wrap_padding=args["use_wrap_padding"], dnet_out_type=args["dnet_out_type"], min_depth=args["min_depth"])


    return model