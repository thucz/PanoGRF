from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

# from .resnet import *
# from .mobilenet import *
from models.layers import Concat, BiProj, CEELayer

from collections import OrderedDict
# from .convert_module import erp_convert
from models.convert_tp.equi2pers_v3 import equi2pers
from models.convert_tp.pers2equi_v3 import pers2equi
import copy
import torch.nn.functional as F
from .ops import conv3x3, conv1x1, conv, upconv,  ResidualBlock, BasicBlock
from models.common_blocks import WrapPadding

from .resnet_convert import convert_to_tp
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# def convert_conv(layer):

#     for name, module in layer.named_modules():
#         if name:
#             try:
#                 # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
#                 sub_layer = getattr(layer, name)
#                 if isinstance(sub_layer, nn.Conv2d):
#                     m = copy.deepcopy(sub_layer)
#                     new_layer = nn.Conv3d(m.in_channels, m.out_channels, kernel_size=(m.kernel_size[0], m.kernel_size[1], 1), 
#                                   stride=(m.stride[0], m.stride[1], 1), padding=(m.padding[0], m.padding[1], 0), padding_mode='zeros', bias=False)
#                     new_layer.weight.data.copy_(m.weight.data.unsqueeze(-1))
#                     if m.bias is not None:
#                         new_layer.bias.data.copy_(m.bias.data)
#                     # first level of current layer or model contains a batch norm --> replacing.
#                     layer._modules[name] = copy.deepcopy(new_layer)
#             except AttributeError:
#                 # go deeper: set name to layer1, getattr will return layer1 --> call this func again
#                 name = name.split('.')[0]
#                 sub_layer = getattr(layer, name)
#                 sub_layer = convert_conv(sub_layer)
#                 layer.__setattr__(name=name, value=sub_layer)
#     return layer


# def convert_bn(layer):
#     for name, module in layer.named_modules():
#         if name:
#             try:
#                 # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
#                 sub_layer = getattr(layer, name)
#                 if isinstance(sub_layer, nn.BatchNorm2d) or isinstance(sub_layer, nn.InstanceNorm2d):
#                     m = copy.deepcopy(sub_layer)
#                     if isinstance(sub_layer, nn.BatchNorm2d):
#                         new_layer = nn.BatchNorm3d(m.num_features)
#                     elif isinstance(sub_layer, nn.InstanceNorm2d):
#                         new_layer = nn.InstanceNorm3d(m.num_features)
#                     new_layer.weight.data.copy_(m.weight.data)
#                     new_layer.bias.data.copy_(m.bias.data)
#                     new_layer.running_mean.data.copy_(m.running_mean.data)
#                     new_layer.running_var.data.copy_(m.running_var.data)
#                     # first level of current layer or model contains a batch norm --> replacing.
#                     layer._modules[name] = copy.deepcopy(new_layer)
#             except AttributeError:
#                 # go deeper: set name to layer1, getattr will return layer1 --> call this func again
#                 name = name.split('.')[0]
#                 sub_layer = getattr(layer, name)
#                 sub_layer = convert_bn(sub_layer)
#                 layer.__setattr__(name=name, value=sub_layer)
#     return layer 

# def convert_to_tp(layer):
#     layer = convert_bn(layer)
#     layer = convert_conv(layer)
#     return layer




# def convert_in(layer):
#     for name, module in layer.named_modules():
#         if name:
#             try:
#                 # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
#                 sub_layer = getattr(layer, name)
#                 if isinstance(sub_layer, nn.InstanceNorm2d):
#                     m = copy.deepcopy(sub_layer)
#                     new_layer = nn.BatchNorm3d(m.num_features)
#                     new_layer.weight.data.copy_(m.weight.data)
#                     new_layer.bias.data.copy_(m.bias.data)
#                     new_layer.running_mean.data.copy_(m.running_mean.data)
#                     new_layer.running_var.data.copy_(m.running_var.data)
#                     # new_layer.track_running_stats.data.copy(m.track_running_stats.data)
                    

#                     # first level of current layer or model contains a batch norm --> replacing.
#                     layer._modules[name] = copy.deepcopy(new_layer)
#             except AttributeError:
#                 # go deeper: set name to layer1, getattr will return layer1 --> call this func again
#                 name = name.split('.')[0]
#                 sub_layer = getattr(layer, name)
#                 sub_layer = convert_in(sub_layer)
#                 layer.__setattr__(name=name, value=sub_layer)
#     return layer 

class ResUNetLight_ERP_TP(nn.Module):
    # def __init__(self, num_layers, equi_h, equi_w, pretrained=False,
    #              fusion_type="cee", se_in_fusion=True, use_wrap_padding=False, nrows=4, npatches=18, patch_size=(128, 128), fov=(80, 80)):
    def __init__(self, in_dim=3, layers=(2, 3, 6, 3), out_dim=32, inplanes=32, use_wrap_padding=False,
                 fusion_type="cee", se_in_fusion=True,  nrows=4, npatches=18, patch_size=(128, 128), fov=(80, 80),
                 autoencoder=False,
    ):
        super(ResUNetLight_ERP_TP, self).__init__()
        # import ipdb;ipdb.set_trace()

        self.nrows = nrows
        self.npatches = npatches
        self.patch_size = patch_size
        self.fov = fov
        self.autoencoder = autoencoder
        # self.num_layers = num_layers
        # self.equi_h = equi_h
        # self.equi_w = equi_w
        # self.cube_h = equi_h//2

        self.fusion_type = fusion_type
        self.se_in_fusion = se_in_fusion

        # # encoder
        # encoder = {2: mobilenet_v2,
        #            18: resnet18,
        #            34: resnet34,
        #            50: resnet50,
        #            101: resnet101,
        #            152: resnet152}
        
        # if num_layers not in encoder:
        #     raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        # self.equi_encoder = encoder[num_layers](pretrained)
        # self.use_wrap_padding = use_wrap_padding
        # if self.use_wrap_padding:
        #     self.equi_encoder = erp_convert(self.equi_encoder)

        # # pretrain_model = torchvision.models.resnet34(pretrained=True)
        # tp_encoder = encoder[num_layers](pretrained)

        # tp_encoder = convert_conv(tp_encoder)
        # tp_encoder = convert_bn(tp_encoder)

        # self.conv1 = tp_encoder.conv1
        # self.bn1 = tp_encoder.bn1
        # self.relu = nn.ReLU(True)
        # self.layer1 = tp_encoder.layer1  #64
        # self.layer2 = tp_encoder.layer2  #128
        # self.layer3 = tp_encoder.layer3  #256
        # self.layer4 = tp_encoder.layer4  #512



        # self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        # if num_layers > 34:
        #     self.num_ch_enc[1:] *= 4

        # if num_layers < 18:
        #     self.num_ch_enc = np.array([16, 24, 32, 96, 320])

        # # decoder
        # self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        # self.equi_dec_convs = OrderedDict()
        # self.c2e = {}

        Fusion_dict = {"cat": Concat,
                       "biproj": BiProj,
                       "cee": CEELayer}
        FusionLayer = Fusion_dict[self.fusion_type]

        # # self.c2e["5"] = Cube2Equirec(self.cube_h // 32, self.equi_h // 32, self.equi_w // 32)

        # self.equi_dec_convs["fusion_5"] = FusionLayer(self.num_ch_enc[4], SE=self.se_in_fusion)
        # self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])

        # # self.c2e["4"] = Cube2Equirec(self.cube_h // 16, self.equi_h // 16, self.equi_w // 16)
        # self.equi_dec_convs["fusion_4"] = FusionLayer(self.num_ch_enc[3], SE=self.se_in_fusion)
        # self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        # self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        # # self.c2e["3"] = Cube2Equirec(self.cube_h // 8, self.equi_h // 8, self.equi_w // 8)
        # self.equi_dec_convs["fusion_3"] = FusionLayer(self.num_ch_enc[2], SE=self.se_in_fusion)
        # self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        # self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        # # self.c2e["2"] = Cube2Equirec(self.cube_h // 4, self.equi_h // 4, self.equi_w // 4)
        # self.equi_dec_convs["fusion_2"] = FusionLayer(self.num_ch_enc[1], SE=self.se_in_fusion)
        # self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        # self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        # # self.c2e["1"] = Cube2Equirec(self.cube_h // 2, self.equi_h // 2, self.equi_w // 2)

        # # self.equi_dec_convs["fusion_1"] = FusionLayer(self.num_ch_enc[0], SE=self.se_in_fusion)
        # # self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        # # self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
        # # self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])
        # # self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)
        # self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
        
        # if self.use_wrap_padding:
        #     self.equi_decoder = erp_convert(self.equi_decoder)


# class ResUNetLight(nn.Module):
    # def __init__(self, in_dim=3, layers=(2, 3, 6, 3), out_dim=32, inplanes=32, use_wrap_padding=False):
    #     super(ResUNetLight, self).__init__()
        # layers = [2, 3, 6, 3]
        norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1        
        block = BasicBlock
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = inplanes
        # self.fuse0_channels = inplanes

        self.groups = 1  # seems useless
        self.base_width = 64  # seems useless
        
        #erp
        self.use_wrap_padding =use_wrap_padding
        if use_wrap_padding:
            self.conv1 = nn.Sequential(
                    WrapPadding(padding=3),
                    nn.Conv2d(in_dim, self.inplanes, kernel_size=7, stride=2, padding=0, bias=False)
                )
        else:
            self.conv1 = nn.Conv2d(in_dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
                               padding_mode='zeros')
        
        self.bn1 = norm_layer(self.inplanes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 32, layers[0], stride=2, use_wrap_padding=use_wrap_padding)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], use_wrap_padding=use_wrap_padding)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], use_wrap_padding=use_wrap_padding)
        #tp
        # self.use_wrap_padding = use_wrap_padding
        # in the modules constructed, these parameters are revised. So they should be re-initialized.
        norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1        
        block = BasicBlock
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = inplanes
        self.groups = 1  # seems useless
        self.base_width = 64  # seems useless
        
        self.tp_conv1 = nn.Conv2d(in_dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
                               padding_mode='zeros')        
        

        self.tp_conv1 = convert_to_tp(self.tp_conv1)

        self.tp_bn1 = norm_layer(self.inplanes, track_running_stats=False, affine=True)        
        self.tp_bn1 = convert_to_tp(self.tp_bn1)
        self.tp_relu = nn.ReLU(inplace=True)
        self.tp_layer1 = self._make_layer(block, 32, layers[0], stride=2)

        self.tp_layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.tp_layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.tp_layer1 = convert_to_tp(self.tp_layer1)
        self.tp_layer2 = convert_to_tp(self.tp_layer2)
        self.tp_layer3 = convert_to_tp(self.tp_layer3)

        # self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        # if num_layers > 34:
        #     self.num_ch_enc[1:] *= 4

        # if num_layers < 18:
            # self.num_ch_enc = np.array([16, 24, 32, 96, 320])
        # self.num_ch_enc = np.array([16, 24, 32, 96, 320])

        # decoder
        self.num_ch_enc = np.array([inplanes, 32, 64, 128, 256])
        self.equi_dec_convs = {}
        self.equi_dec_convs["fusion_1"] = FusionLayer(self.num_ch_enc[0], SE=self.se_in_fusion).cuda()
        self.equi_dec_convs["fusion_2"] = FusionLayer(self.num_ch_enc[1], SE=self.se_in_fusion).cuda()
        self.equi_dec_convs["fusion_3"] = FusionLayer(self.num_ch_enc[2], SE=self.se_in_fusion).cuda()
        self.equi_dec_convs["fusion_4"] = FusionLayer(self.num_ch_enc[3], SE=self.se_in_fusion).cuda()
        
        # decoder
        self.upconv3 = upconv(128, 64, 3, 2, use_wrap_padding=use_wrap_padding)
        self.iconv3 = conv(64 + 64, 64, 3, 1, use_wrap_padding=use_wrap_padding)
        self.recons_conv3 = nn.Conv2d(64, 3, 1, 1)


        self.upconv2 = upconv(64, 32, 3, 2, use_wrap_padding=use_wrap_padding)
        self.iconv2 = conv(32 + 32, 32, 3, 1, use_wrap_padding=use_wrap_padding)
        self.recons_conv2 = nn.Conv2d(32, 3, 1, 1)

        # fine-level conv
        self.out_conv = nn.Conv2d(32, out_dim, 1, 1)

        # #1/2
        self.upconv1 = upconv(32, inplanes, 3, 2, use_wrap_padding=use_wrap_padding)
        self.iconv1 = conv(inplanes*2, inplanes, 3, 1, use_wrap_padding=use_wrap_padding)
        self.recons_conv1 = nn.Conv2d(inplanes, 3, 1, 1)
        
        # orig
        self.upconv0 = upconv(inplanes, inplanes, 3, 2, use_wrap_padding=use_wrap_padding)
        self.recons_conv0 = nn.Conv2d(inplanes, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()


        # #original res
        # equi_x = self.upconv0(equi_x)
        # img_out = self.sigmoid(self.recons_conv(equi_x))


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_wrap_padding=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, use_wrap_padding=use_wrap_padding),
                norm_layer(planes * block.expansion, track_running_stats=False, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, use_wrap_padding=use_wrap_padding))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, use_wrap_padding=use_wrap_padding))

        return nn.Sequential(*layers)

    def skipconnect(self, x1, x2):
        #x2.size>x1.size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if self.use_wrap_padding:
            x1 = F.pad(x1, (0, 0, diffY // 2, diffY - diffY // 2), mode = 'constant')
            x1 = F.pad(x1, ( diffX // 2,  diffX - diffX // 2, 0, 0), mode = 'circular')
        else:
            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        #tangent projection:
        bs, _, erp_h, erp_w = x.shape
        # print("x.shape:", x.shape)
        patch_h, patch_w = pair(self.patch_size)
        # import ipdb;ipdb.set_trace()
        x_tp, _, _, _ = equi2pers(x, self.fov, self.nrows, patch_size=self.patch_size)            

        x = self.relu(self.bn1(self.conv1(x)))#1/2
        x1 = self.layer1(x)#1/4
        x2 = self.layer2(x1)#1/8
        x3 = self.layer3(x2)#1/16

        #tp
        
        x_tp = self.relu(self.tp_bn1(self.tp_conv1(x_tp))) #->(64, 64, 18) 1/2

        x_tp_1 = self.tp_layer1(x_tp)#->(32, 32, 18)#1/4
        x_tp_2 = self.tp_layer2(x_tp_1)#->(16, 16, 18)#1/8
        x_tp_3 = self.tp_layer3(x_tp_2)#->(8, 8, 18)#1/16
        # import ipdb;ipdb.set_trace()
        #1/16
        t2e_enc_feat3 = pers2equi(x_tp_3, self.fov, self.nrows, (patch_h//16, patch_w//16), (erp_h//16, erp_w//16), 'ResUnetLight_3') 
        fused_feat3 = self.equi_dec_convs["fusion_4"](x3, t2e_enc_feat3)

        #1/8
        equi_x = self.upconv3(fused_feat3)
        t2e_enc_feat2 = pers2equi(x_tp_2, self.fov, self.nrows, (patch_h//8, patch_w//8), (erp_h//8, erp_w//8), 'ResUnetLight_2') 
        fused_feat2 = self.equi_dec_convs["fusion_3"](x2, t2e_enc_feat2)       
        equi_x = self.skipconnect(fused_feat2, equi_x)        
        equi_x = self.iconv3(equi_x)
        outputs = {}
        outputs[("pred_img", 3)] = self.sigmoid(self.recons_conv3(equi_x))

        #1/4
        equi_x = self.upconv2(equi_x)#orig
        t2e_enc_feat1 = pers2equi(x_tp_1, self.fov, self.nrows, (patch_h//4, patch_w//4), (erp_h//4, erp_w//4), 'ResUnetLight_1')
        fused_feat1 = self.equi_dec_convs["fusion_2"](x1, t2e_enc_feat1) 
        equi_x = self.skipconnect(fused_feat1, equi_x)#orig
        equi_x = self.iconv2(equi_x)
        
        x_out = self.out_conv(equi_x)
        if self.autoencoder:
            outputs[("pred_img", 2)] = self.sigmoid(self.recons_conv2(equi_x))

            #1/2
            equi_x = self.upconv1(equi_x)
            t2e_enc_feat0 = pers2equi(x_tp, self.fov, self.nrows, (patch_h//2 , patch_w//2), (erp_h//2, erp_w//2), 'ResUnetLight_0')
            # import ipdb;ipdb.set_trace()
            
            fused_feat0 = self.equi_dec_convs["fusion_1"](x, t2e_enc_feat0)
            equi_x = self.skipconnect(fused_feat0, equi_x)#orig
            equi_x = self.iconv1(equi_x)
            outputs[("pred_img", 1)] = self.sigmoid(self.recons_conv1(equi_x))

            #original res
            equi_x = self.upconv0(equi_x)
            # img_out = self.sigmoid(self.recons_conv(equi_x))        

            outputs[("pred_img", 0)] = self.sigmoid(self.recons_conv0(equi_x))

            # import ipdb;ipdb.set_trace()
            # print("x_out.shape:", x_out.shape)

            # # disp
            # self.disp4 = Conv3x3(num_ch_dec[3], num_output_channels)
            # self.disp3 = Conv3x3(num_ch_dec[2], num_output_channels)
            # self.disp2 = Conv3x3(num_ch_dec[1], num_output_channels)
            # self.disp1 = Conv3x3(num_ch_dec[0], num_output_channels)

            # return self.outputs

            #deconv
            # if self.autoencoder:
            return x_out, outputs
            

        return x_out

    # def forward(self, input_equi_image, input_cube_image):
    #     if self.num_layers < 18:
    #         equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4 \
    #             = self.equi_encoder(input_equi_image)

    #     else:            
    #         x = self.equi_encoder.conv1(input_equi_image)
    #         x = self.equi_encoder.relu(self.equi_encoder.bn1(x))
    #         equi_enc_feat0 = x

    #         x = self.equi_encoder.maxpool(x)
    #         equi_enc_feat1 = self.equi_encoder.layer1(x)
    #         equi_enc_feat2 = self.equi_encoder.layer2(equi_enc_feat1)
    #         equi_enc_feat3 = self.equi_encoder.layer3(equi_enc_feat2)
    #         equi_enc_feat4 = self.equi_encoder.layer4(equi_enc_feat3)

    #     #tangent projection:
    #     bs, _, erp_h, erp_w = input_equi_image.shape
    #     # device = rgb.device
    #     patch_h, patch_w = pair(self.patch_size)
    #     high_res_patch, _, _, _ = equi2pers(input_equi_image, self.fov, self.nrows, patch_size=self.patch_size)    
    #     conv1 = self.relu(self.bn1(self.conv1(high_res_patch)))
    #     pool = F.max_pool3d(conv1, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

    #     layer1 = self.layer1(pool)
    #     layer2 = self.layer2(layer1)
    #     layer3 = self.layer3(layer2)
    #     layer4 = self.layer4(layer3)
    #     # layer4_reshape = self.down(layer4)
    #     # layer4_reshape = layer4_reshape.reshape(bs, -1, n_patch).transpose(1, 2)

    #     # layer4_reshape = self.transformer(layer4_reshape)
    #     # layer4_reshape = layer4_reshape.transpose(1, 2).reshape(bs, -1, 1, 1, n_patch)
    #     # layer4 = layer4 + layer4_reshape
    #     # print("layer4.shape:", layer4.shape)

    #     # equi image decoding fused with tangent projection features
    #     outputs = {}

    #     # tp_enc_feat4 = torch.cat(torch.split(cube_enc_feat4, input_equi_image.shape[0], dim=0), dim=-1)
        
    #     # c2e_enc_feat4 = self.c2e["5"](cube_enc_feat4)
    #     #1/32:layer = bs, c, h, w, n_patch
    #     t2e_enc_feat4 = pers2equi(layer4, self.fov, self.nrows, (patch_h//32, patch_w//32), (erp_h//32, erp_w//32), 'layer4') 

    #     fused_feat4 = self.equi_dec_convs["fusion_5"](equi_enc_feat4, t2e_enc_feat4)
    #     equi_x = upsample(self.equi_dec_convs["upconv_5"](fused_feat4))


    #     # .permute(0, 2, 3, 4, 1)
    #     #bs, c, h, w, n_patch


    #     #1/32:layer = bs, c, h, w, n_patch
    #     t2e_enc_feat3 = pers2equi(layer3, self.fov, self.nrows, (patch_h//16, patch_w//16), (erp_h//16, erp_w//16), 'layer3') 

    #     fused_feat3 = self.equi_dec_convs["fusion_4"](equi_enc_feat3, t2e_enc_feat3)
    #     equi_x = torch.cat([equi_x, fused_feat3], 1)

    #     equi_x = self.equi_dec_convs["deconv_4"](equi_x)
    #     equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))
        
    #     # tp_enc_feat2 = torch.cat(torch.split(cube_enc_feat2, input_equi_image.shape[0], dim=0), dim=-1)

    #     # t2e_enc_feat2 = self.c2e["3"](cube_enc_feat2)
    #     t2e_enc_feat2 = pers2equi(layer2, self.fov, self.nrows, (patch_h//8, patch_w//8), (erp_h//8, erp_w//8), 'layer2') 
    #     fused_feat2 = self.equi_dec_convs["fusion_3"](equi_enc_feat2, t2e_enc_feat2)
    #     equi_x = torch.cat([equi_x, fused_feat2], 1)
    #     equi_x = self.equi_dec_convs["deconv_3"](equi_x)
    #     equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))

    #     # cube_enc_feat1 = torch.cat(torch.split(cube_enc_feat1, input_equi_image.shape[0], dim=0), dim=-1)
    #     # c2e_enc_feat1 = self.c2e["2"](cube_enc_feat1)
    #     t2e_enc_feat1 = pers2equi(layer1, self.fov, self.nrows, (patch_h//4, patch_w//4), (erp_h//4, erp_w//4), 'layer1') 
        
    #     fused_feat1 = self.equi_dec_convs["fusion_2"](equi_enc_feat1, t2e_enc_feat1)
    #     equi_x = torch.cat([equi_x, fused_feat1], 1)
    #     equi_x = self.equi_dec_convs["deconv_2"](equi_x)         
    #     # mono_feat = equi_x#1/4
    #     # print("mono_feat.shape:", mono_feat.shape)
    #     return self.equi_dec_convs["upconv_2"](equi_x) #1/4
