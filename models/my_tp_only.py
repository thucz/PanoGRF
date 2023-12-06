from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from .resnet import *
from .mobilenet import *
from .layers import Conv3x3, ConvBlock, upsample, Cube2Equirec, Concat, BiProj, CEELayer

from collections import OrderedDict

from .convert_module import erp_convert
from models.convert_tp.equi2pers_v3 import equi2pers
from models.convert_tp.pers2equi_v3 import pers2equi
import copy
import torch.nn.functional as F
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
def convert_conv(layer):
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, nn.Conv2d):
                    m = copy.deepcopy(sub_layer)
                    new_layer = nn.Conv3d(m.in_channels, m.out_channels, kernel_size=(m.kernel_size[0], m.kernel_size[1], 1), 
                                  stride=(m.stride[0], m.stride[1], 1), padding=(m.padding[0], m.padding[1], 0), padding_mode='zeros', bias=False)
                    new_layer.weight.data.copy_(m.weight.data.unsqueeze(-1))
                    if m.bias is not None:
                        new_layer.bias.data.copy_(m.bias.data)
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = copy.deepcopy(new_layer)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = convert_conv(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer


def convert_bn(layer):
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, nn.BatchNorm2d):
                    m = copy.deepcopy(sub_layer)
                    new_layer = nn.BatchNorm3d(m.num_features)
                    new_layer.weight.data.copy_(m.weight.data)
                    new_layer.bias.data.copy_(m.bias.data)
                    new_layer.running_mean.data.copy_(m.running_mean.data)
                    new_layer.running_var.data.copy_(m.running_var.data)
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = copy.deepcopy(new_layer)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(layer, name)
                sub_layer = convert_bn(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer 
class TP(nn.Module):
    def __init__(self, num_layers, equi_h, equi_w, pretrained=False,
                 fusion_type="cee", se_in_fusion=True, use_wrap_padding=False, nrows=4, npatches=18, patch_size=(128, 128), fov=(80, 80)):
        self.nrows = nrows
        self.npatches = npatches
        self.patch_size = patch_size
        self.fov = fov
        super(TP, self).__init__()
        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w
        # self.cube_h = equi_h//2

        self.fusion_type = fusion_type
        self.se_in_fusion = se_in_fusion

        # encoder
        encoder = {2: mobilenet_v2,
                   18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,
                   152: resnet152}
        if num_layers not in encoder:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        # self.equi_encoder = encoder[num_layers](pretrained)
        self.use_wrap_padding = use_wrap_padding
        # if self.use_wrap_padding:
        #     self.equi_encoder = erp_convert(self.equi_encoder)

        # pretrain_model = torchvision.models.resnet34(pretrained=True)
        tp_encoder = encoder[num_layers](pretrained)

        tp_encoder = convert_conv(tp_encoder)
        tp_encoder = convert_bn(tp_encoder)

        self.conv1 = tp_encoder.conv1
        self.bn1 = tp_encoder.bn1
        self.relu = nn.ReLU(True)
        self.layer1 = tp_encoder.layer1  #64
        self.layer2 = tp_encoder.layer2  #128
        self.layer3 = tp_encoder.layer3  #256
        self.layer4 = tp_encoder.layer4  #512



        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if num_layers < 18:
            self.num_ch_enc = np.array([16, 24, 32, 96, 320])

        # decoder
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.equi_dec_convs = OrderedDict()
        self.c2e = {}

        # Fusion_dict = {"cat": Concat,
        #                "biproj": BiProj,
        #                "cee": CEELayer}
        # FusionLayer = Fusion_dict[self.fusion_type]

        # self.c2e["5"] = Cube2Equirec(self.cube_h // 32, self.equi_h // 32, self.equi_w // 32)

        # self.equi_dec_convs["fusion_5"] = FusionLayer(self.num_ch_enc[4], SE=self.se_in_fusion)
        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])

        # self.c2e["4"] = Cube2Equirec(self.cube_h // 16, self.equi_h // 16, self.equi_w // 16)
        # self.equi_dec_convs["fusion_4"] = FusionLayer(self.num_ch_enc[3], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        # self.c2e["3"] = Cube2Equirec(self.cube_h // 8, self.equi_h // 8, self.equi_w // 8)
        # self.equi_dec_convs["fusion_3"] = FusionLayer(self.num_ch_enc[2], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        # self.c2e["2"] = Cube2Equirec(self.cube_h // 4, self.equi_h // 4, self.equi_w // 4)
        # self.equi_dec_convs["fusion_2"] = FusionLayer(self.num_ch_enc[1], SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        # self.c2e["1"] = Cube2Equirec(self.cube_h // 2, self.equi_h // 2, self.equi_w // 2)
        # self.equi_dec_convs["fusion_1"] = FusionLayer(self.num_ch_enc[0], SE=self.se_in_fusion)
        # self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        # self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])

        # self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])

        # self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
        
        if self.use_wrap_padding:
            self.equi_decoder = erp_convert(self.equi_decoder)


        # self.down = nn.Conv3d(512, 512//16, kernel_size=1, stride=1, padding=0)
        # self.transformer = Transformer_cascade(512, npatches, depth=6, num_heads=4)
        
        # self.de_conv0_0 = ConvBnReLU_v2(512, 256, kernel_size=3, stride=1)
        # self.de_conv0_1 = ConvBnReLU_v2(256+256, 128, kernel_size=3, stride=1) 
        # self.de_conv1_0 = ConvBnReLU_v2(128, 128, kernel_size=3, stride=1)
        # self.de_conv1_1 = ConvBnReLU_v2(128+128, 64, kernel_size=3, stride=1)
        # self.de_conv2_0 = ConvBnReLU_v2(64, 64, kernel_size=3, stride=1)
        # self.de_conv2_1 = ConvBnReLU_v2(64+64, 64, kernel_size=3, stride=1)
        # self.de_conv3_0 = ConvBnReLU_v2(64, 64, kernel_size=3, stride=1)
        # self.de_conv3_1 = ConvBnReLU_v2(64+64, 32, kernel_size=3, stride=1)
        # self.de_conv4_0 = ConvBnReLU_v2(32, 32, kernel_size=3, stride=1)
        # self.pred = nn.Conv3d(32, 1, (3, 3, 1), 1, padding=(1, 1, 0), padding_mode='zeros')
        # self.weight_pred = nn.Conv3d(32, 1, (3, 3, 1), 1, padding=(1, 1, 0), padding_mode='zeros')
        # self.min_depth = 0.1
        # self.max_depth = 8.0

        # self.mlp_points = nn.Sequential(
        #         nn.Conv2d(5, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(16),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(inplace=True),
        # )

    
    def forward(self, input_equi_image, input_cube_image):
        # if self.num_layers < 18:
        #     equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4 \
        #         = self.equi_encoder(input_equi_image)

        # else:            
        #     x = self.equi_encoder.conv1(input_equi_image)
        #     x = self.equi_encoder.relu(self.equi_encoder.bn1(x))
        #     equi_enc_feat0 = x

        #     x = self.equi_encoder.maxpool(x)
        #     equi_enc_feat1 = self.equi_encoder.layer1(x)
        #     equi_enc_feat2 = self.equi_encoder.layer2(equi_enc_feat1)
        #     equi_enc_feat3 = self.equi_encoder.layer3(equi_enc_feat2)
        #     equi_enc_feat4 = self.equi_encoder.layer4(equi_enc_feat3)

        #tangent projection:
        bs, _, erp_h, erp_w = input_equi_image.shape
        # device = rgb.device
        patch_h, patch_w = pair(self.patch_size)
        high_res_patch, _, _, _ = equi2pers(input_equi_image, self.fov, self.nrows, patch_size=self.patch_size)    
        conv1 = self.relu(self.bn1(self.conv1(high_res_patch)))
        pool = F.max_pool3d(conv1, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        layer1 = self.layer1(pool)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # layer4_reshape = self.down(layer4)
        # layer4_reshape = layer4_reshape.reshape(bs, -1, n_patch).transpose(1, 2)

        # layer4_reshape = self.transformer(layer4_reshape)
        # layer4_reshape = layer4_reshape.transpose(1, 2).reshape(bs, -1, 1, 1, n_patch)
        # layer4 = layer4 + layer4_reshape
        # print("layer4.shape:", layer4.shape)


        # equi image decoding fused with tangent projection features
        outputs = {}

        # tp_enc_feat4 = torch.cat(torch.split(cube_enc_feat4, input_equi_image.shape[0], dim=0), dim=-1)
        
        # c2e_enc_feat4 = self.c2e["5"](cube_enc_feat4)
        #1/32:layer = bs, c, h, w, n_patch
        t2e_enc_feat4 = pers2equi(layer4, self.fov, self.nrows, (patch_h//32, patch_w//32), (erp_h//32, erp_w//32), 'layer4')

        # fused_feat4 = self.equi_dec_convs["fusion_5"](equi_enc_feat4, t2e_enc_feat4)
        equi_x = upsample(self.equi_dec_convs["upconv_5"](t2e_enc_feat4))


        # .permute(0, 2, 3, 4, 1)
        #bs, c, h, w, n_patch


        #1/32:layer = bs, c, h, w, n_patch
        t2e_enc_feat3 = pers2equi(layer3, self.fov, self.nrows, (patch_h//16, patch_w//16), (erp_h//16, erp_w//16), 'layer3') 

        # fused_feat3 = self.equi_dec_convs["fusion_4"](equi_enc_feat3, t2e_enc_feat3)
        equi_x = torch.cat([equi_x, t2e_enc_feat3], 1)
        equi_x = self.equi_dec_convs["deconv_4"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))
        
        # tp_enc_feat2 = torch.cat(torch.split(cube_enc_feat2, input_equi_image.shape[0], dim=0), dim=-1)

        # t2e_enc_feat2 = self.c2e["3"](cube_enc_feat2)
        t2e_enc_feat2 = pers2equi(layer2, self.fov, self.nrows, (patch_h//8, patch_w//8), (erp_h//8, erp_w//8), 'layer2') 
        # fused_feat2 = self.equi_dec_convs["fusion_3"](equi_enc_feat2, t2e_enc_feat2)
        equi_x = torch.cat([equi_x, t2e_enc_feat2], 1)
        equi_x = self.equi_dec_convs["deconv_3"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))

        # cube_enc_feat1 = torch.cat(torch.split(cube_enc_feat1, input_equi_image.shape[0], dim=0), dim=-1)
        # c2e_enc_feat1 = self.c2e["2"](cube_enc_feat1)
        t2e_enc_feat1 = pers2equi(layer1, self.fov, self.nrows, (patch_h//4, patch_w//4), (erp_h//4, erp_w//4), 'layer1') 
        
        # fused_feat1 = self.equi_dec_convs["fusion_2"](equi_enc_feat1, t2e_enc_feat1)
        equi_x = torch.cat([equi_x, t2e_enc_feat1], 1)
        equi_x = self.equi_dec_convs["deconv_2"](equi_x)         
        # mono_feat = equi_x#1/4
        # print("mono_feat.shape:", mono_feat.shape)
        return self.equi_dec_convs["upconv_2"](equi_x) #1/4
