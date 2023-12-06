import torch.nn as nn
from .common_blocks import WrapPadding3D
from inplace_abn import InPlaceABN
import torch
# class ConvBnReLU3D(nn.Module):
#     def __init__(self, in_channels, out_channels,
#                  kernel_size=3, stride=1, pad=1,
#                  norm_act=InPlaceABN):
#         super(ConvBnReLU3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels,
#                               kernel_size, stride=stride, padding=pad, bias=False)
#         self.bn = norm_act(out_channels)
    # def forward(self, x):
    #     return self.bn(self.conv(x))
class ConvBnReLU3DWrap(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, use_wrap_padding=True, use_v_input=False, norm_act=None):
        super(ConvBnReLU3DWrap, self).__init__()
        if use_wrap_padding:
            self.padding = WrapPadding3D(padding=(padding, padding, padding))
        else:
            self.padding = nn.Identity()
        
        self.conv = nn.Conv3d(in_channels + (1 if use_v_input else 0), out_channels,
                              kernel_size, stride=stride, padding=0 if use_wrap_padding else padding, bias=False, padding_mode="zeros")
        if norm_act is not None:
            self.norm_act = norm_act
        else:
            self.bn = nn.BatchNorm3d(num_features=out_channels)        
            self.act = nn.LeakyReLU()
            self.norm_act = None
        
    
    def forward(self, x):
        x = self.padding(x)
    
        if self.norm_act is not None:
            return self.norm_act(self.conv(x))
        else:
            return self.act(self.bn(self.conv(x)))


class Conv3DWrap(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, use_wrap_padding=True, use_v_input=False):
        super(Conv3DWrap, self).__init__()
        if use_wrap_padding:
            self.padding = WrapPadding3D(padding=(padding, padding, padding))
        else:
            self.padding = nn.Identity()
        
        self.conv = nn.Conv3d(in_channels + (1 if use_v_input else 0), out_channels,
                              kernel_size, stride=stride, padding=0 if use_wrap_padding else padding, bias=False, padding_mode="zeros")
    def forward(self, x):
        x = self.padding(x)

        return self.conv(x)
class UpConv3DWrap(nn.Module):
    # self.conv7 = UpConv3DWrap(64, 32, 3, padding=1, output_padding=1,
    #                     stride=2, bias=False, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act(32))

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, output_padding=1, bias=False, use_wrap_padding=True, use_v_input=False, norm_act=None):
        super(UpConv3DWrap, self).__init__()

        if use_wrap_padding:
            self.padding = WrapPadding3D(padding=(padding, padding, padding))
        else:
            self.padding = nn.Identity()
        # self.tconv = nn.ConvTranspose3d(in_channels + (1 if use_v_input else 0), out_channels, kernel_size, padding=0 if use_wrap_padding else padding, output_padding=0 if use_wrap_padding else output_padding, padding_mode="zeros",
        #                        stride=stride, bias=bias)
        self.up_scale=stride

        self.conv = nn.Conv3d(in_channels + (1 if use_v_input else 0), out_channels,
                              kernel_size, stride=1, padding=0 if use_wrap_padding else padding, bias=bias, padding_mode="zeros")

            

        if norm_act is not None:
            self.norm_act = norm_act
        else:
            self.bn = nn.BatchNorm3d(num_features=out_channels)        
            self.act = nn.LeakyReLU()
            self.norm_act = None
    def forward(self, x):
        x = torch.nn.functional.interpolate(x,
                                          scale_factor=self.up_scale,
                                          mode="trilinear",
                                          align_corners=False)
        x = self.padding(x)
        if self.norm_act is not None:
            return self.norm_act(self.conv(x))
        else:
            return self.act(self.bn(self.conv(x)))
    


    
class CostRegNet(nn.Module):
    def __init__(self, cfg, in_channels):
        self.cfg = cfg
        super(CostRegNet, self).__init__()
        # Conv3DBlockv2(in_channels=channels,
        #               out_channels=2 * channels,
        #               kernel_size=(3, 3, 3),
        #               stride=(1, 1, 1),
        #               padding=(1, 1, 1),
        #               use_batch_norm=False,
        #               use_wrap_padding=use_wrap_padding,
        #               use_v_input=use_v_input)
        if self.cfg["inplace_abn"]:
            norm_act = InPlaceABN
        else:
            norm_act = None
        # use_wrap_padding=True, use_v_input=False
        self.conv0 = ConvBnReLU3DWrap(in_channels, 8, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act)
        self.conv1 = ConvBnReLU3DWrap(8, 16, stride=2, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act)
        self.conv2 = ConvBnReLU3DWrap(16, 16, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act)
        self.conv3 = ConvBnReLU3DWrap(16, 32, stride=2, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act)
        self.conv4 = ConvBnReLU3DWrap(32, 32, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act)
        self.conv5 = ConvBnReLU3DWrap(32, 64, stride=2, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act)
        self.conv6 = ConvBnReLU3DWrap(64, 64, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act)

        # self.conv7 = nn.Sequential(
        #     nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(32))

        # self.conv9 = nn.Sequential(
        #     nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(16))

        # self.conv11 = nn.Sequential(
        #     nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(8))
        if norm_act is not None:
            self.conv7 = UpConv3DWrap(64, 32, 3, padding=1,
                                stride=2, bias=False, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act(32))

            self.conv9 = UpConv3DWrap(32, 16, 3, padding=1,
                                stride=2, bias=False, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act(16))

            self.conv11 = UpConv3DWrap(16, 8, 3, padding=1,
                                stride=2, bias=False, use_wrap_padding=cfg["use_wrap_padding"], norm_act=norm_act(8))
        else:
            self.conv7 = UpConv3DWrap(64, 32, 3, padding=1,
                                stride=2, bias=False, use_wrap_padding=cfg["use_wrap_padding"])

            self.conv9 = UpConv3DWrap(32, 16, 3, padding=1,
                                stride=2, bias=False, use_wrap_padding=cfg["use_wrap_padding"])#, norm_act=norm_act(16))

            self.conv11 = UpConv3DWrap(16, 8, 3, padding=1,
                                stride=2, bias=False, use_wrap_padding=cfg["use_wrap_padding"])#, norm_act=norm_act(8))
            

        # self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)
        self.prob = Conv3DWrap(8, 1, 3, stride=1, padding=1, use_wrap_padding=cfg["use_wrap_padding"])


    def forward(self, x):
        # print('x.shape:', x.shape) #2, 8, 64, 128, 256
        conv0 = self.conv0(x)#2, 8, 64, 128, 256
        conv2 = self.conv2(self.conv1(conv0))#2, 16, 32, 64, 128
        conv4 = self.conv4(self.conv3(conv2))#2, 32, 16, 32, 64
        x = self.conv6(self.conv5(conv4))#x.shape: 2, 64, 8, 16, 32
        x = conv4 + self.conv7(x) #
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.prob(x)
        return x