import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common_blocks import WrapPadding
import math
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, use_wrap_padding=False):
    """3x3 convolution with padding"""
    if use_wrap_padding:
        return nn.Sequential(
            WrapPadding(padding=dilation),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)
        )
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='zeros')


def conv1x1(in_planes, out_planes, stride=1, use_wrap_padding=False):
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='zeros')
    # if handle_distort:
    #     in_dim = in_planes + 1
    # else:
    in_dim = in_planes
    if use_wrap_padding:
        return nn.Conv2d(in_dim, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
        return nn.Conv2d(in_dim, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='zeros')


def interpolate_feats(feats, points, h=None, w=None, padding_mode='zeros', align_corners=False, inter_mode='bilinear'):
    """

    :param feats:   b,f,h,w
    :param points:  b,n,2
    :param h:       float
    :param w:       float
    :param padding_mode:
    :param align_corners:
    :param inter_mode:
    :return:
    """
    b, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw    
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)    # [srn,1,n,2]
    feats_inter = F.grid_sample(feats, points_norm, mode=inter_mode, padding_mode=padding_mode, align_corners=align_corners).squeeze(2)      # srn,f,n    
    feats_inter = feats_inter.permute(0,2,1)#srn, n, f
    return  feats_inter

def masked_mean_var(feats,mask,dim=2):
    mask=mask.float() # b,1,n,1
    mask_sum = torch.clamp_min(torch.sum(mask,dim,keepdim=True),min=1e-4) # b,1,1,1
    feats_mean = torch.sum(feats*mask,dim,keepdim=True)/mask_sum  # b,f,1,1
    feats_var = torch.sum((feats-feats_mean)**2*mask,dim,keepdim=True)/mask_sum # b,f,1,1
    return feats_mean, feats_var

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inter=None, use_norm=True, norm_layer=nn.BatchNorm2d,bias=False, use_wrap_padding=False):
        super().__init__()
        if dim_inter is None:
            dim_inter=dim_out

        if use_norm:
            if use_wrap_padding: 
                self.conv=nn.Sequential(
                    norm_layer(dim_in),
                    nn.ReLU(True),
                    WrapPadding(padding=1),
                    nn.Conv2d(dim_in,dim_inter,3,1,padding=0,bias=bias),
                    norm_layer(dim_inter),
                    nn.ReLU(True),
                    WrapPadding(padding=1),                    
                    nn.Conv2d(dim_inter,dim_out,3,1,padding=0,bias=bias),
                )
            else:               
                self.conv=nn.Sequential(
                    norm_layer(dim_in),
                    nn.ReLU(True),
                    nn.Conv2d(dim_in,dim_inter,3,1,1,bias=bias,padding_mode='zeros'),
                    norm_layer(dim_inter),
                    nn.ReLU(True),
                    nn.Conv2d(dim_inter,dim_out,3,1,1,bias=bias,padding_mode='zeros'),
                )
        else:
            if use_wrap_padding:
                self.conv=nn.Sequential(
                    nn.ReLU(True),
                    WrapPadding(padding=1),
                    nn.Conv2d(dim_in,dim_inter,3,1,padding=0),
                    nn.ReLU(True),
                    WrapPadding(padding=1),
                    nn.Conv2d(dim_inter,dim_out,3,1,padding=0),
                )

            else:
                self.conv=nn.Sequential(
                    nn.ReLU(True),
                    nn.Conv2d(dim_in,dim_inter,3,1,1),
                    nn.ReLU(True),
                    nn.Conv2d(dim_inter,dim_out,3,1,1),
                )

        self.short_cut=None
        if dim_in!=dim_out:
            self.short_cut=nn.Conv2d(dim_in,dim_out,1,1)

    def forward(self, feats):
        feats_out=self.conv(feats)
        if self.short_cut is not None:
            feats_out=self.short_cut(feats)+feats_out
        else:
            feats_out=feats_out+feats
        return feats_out

class AddBias(nn.Module):
    def __init__(self,val):
        super().__init__()
        self.val=val

    def forward(self,x):
        return x+self.val

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_wrap_padding=False, handle_distort=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # sin_phi = torch.arange(0, height//4, dtype=torch.float32).cuda()
        # sin_phi = torch.sin((sin_phi + 0.5) * math.pi / (height//4))
        # sin_phi = sin_phi.view(1, 1, height//4, 1).expand(batch_size, 1, height//4, width//4)
        # self.sin_phi = sin_phi
        self.handle_distort=handle_distort
        if handle_distort:
            conv1_dim = inplanes+1
        else:
            conv1_dim = inplanes

        if handle_distort:
            conv2_dim = planes+1
        else:
            conv2_dim = planes
        self.conv1 = conv3x3(conv1_dim, planes, stride, use_wrap_padding=use_wrap_padding)

        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(conv2_dim, planes, use_wrap_padding=use_wrap_padding)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride
    def expand_sin(self, x):
        if self.handle_distort:
            bs, _, height, width = x.shape
            sin_phi = torch.arange(0, height, dtype=torch.float32).cuda()
            sin_phi = torch.sin((sin_phi + 0.5) * math.pi / (height))
            sin_phi = sin_phi.view(1, 1, height, 1).expand(bs, 1, height, width)
            x = torch.cat([x, sin_phi],dim=1)  
        return x      

    def forward(self, x):
        identity = x
        # if self.handle_distort:
        #     bs, _, height, width = x.shape
        #     sin_phi = torch.arange(0, height, dtype=torch.float32).cuda()
        #     sin_phi = torch.sin((sin_phi + 0.5) * math.pi / (height))
        #     sin_phi = sin_phi.view(1, 1, height, 1).expand(bs, 1, height, width)
        #     x = torch.cat([x, sin_phi],dim=1)        
        out = self.conv1(self.expand_sin(x))
        out = self.bn1(out)
        out = self.relu(out)

        # if self.handle_distort:
        #     # import ipdb;ipdb.set_trace()
        #     out = torch.cat([out, sin_phi],dim=1)        


        out = self.conv2(self.expand_sin(out))
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(self.expand_sin(x)) #

        out += identity
        out = self.relu(out)

        return out

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride, use_wrap_padding=False):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        if use_wrap_padding:
            self.conv = nn.Sequential(
                WrapPadding(padding=(self.kernel_size - 1) // 2),
                nn.Conv2d(num_in_layers,
                                num_out_layers,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=0
                                )
                )
        else:
            self.conv = nn.Conv2d(num_in_layers,
                                num_out_layers,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=(self.kernel_size - 1) // 2,
                                padding_mode='zeros')
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)

class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale, use_wrap_padding=False):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)

class ResUNetLight(nn.Module):
    def __init__(self, cfg, in_dim=3, layers=(2, 3, 6, 3), out_dim=32, inplanes=32, use_wrap_padding=False, autoencoder=False):
        super(ResUNetLight, self).__init__()
        self.cfg = cfg
        # layers = [2, 3, 6, 3]
        norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.autoencoder = autoencoder
        # if self.cfg["handle_distort_input_all"]:
        handle_distort=self.cfg["handle_distort_input_all"]
        self.handle_distort = handle_distort
        block = BasicBlock
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = inplanes
        self.groups = 1  # seems useless
        self.base_width = 64  # seems useless
        self.use_wrap_padding =use_wrap_padding
        if self.cfg["handle_distort"]:
        #     # Polar Branch
        #     # self.polarcoord = nn.Sequential(
        #     #     conv3x3(1, 32, use_wrap_padding=use_wrap_padding),
        #     #     ResidualBlock(32, 32, norm_layer=norm_layer, use_wrap_padding=use_wrap_padding),
        #     #     conv1x1(32, 32, use_wrap_padding=use_wrap_padding),
        #     # )
        #     # bs, _, fh, fw = volume_feats.shape
        #     height= self.cfg["height"]
        #     width = self.cfg["width"]
        #     batch_size = 2 #self.cfg["batch_size"]

        #     sin_phi = torch.arange(0, height, dtype=torch.float32).cuda()
        #     sin_phi = torch.sin((sin_phi + 0.5) * math.pi / height)
        #     sin_phi = sin_phi.view(1, 1, height, 1).expand(1, 1, height, width)

        #     self.sin_phi = sin_phi
            in_dim += 1

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
        self.layer1 = self._make_layer(block, 32, layers[0], stride=2, use_wrap_padding=use_wrap_padding, handle_distort=handle_distort)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], use_wrap_padding=use_wrap_padding, handle_distort=handle_distort)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], use_wrap_padding=use_wrap_padding, handle_distort=handle_distort)

        # decoder
        if self.handle_distort:
            up1_dim=128+1
            iconv1_dim = 64 + 64 + 1
            up2_dim = 64+1
            iconv2_dim = 64 +1
            outconv_dim=32+1
        else:
            up1_dim = 128
            up2_dim = 64
            iconv1_dim = 64 + 64
            iconv2_dim = 64
            outconv_dim=32


        self.upconv3 = upconv(up1_dim, 64, 3, 2, use_wrap_padding=use_wrap_padding)

        self.iconv3 = conv(iconv1_dim, 64, 3, 1, use_wrap_padding=use_wrap_padding)
        if self.autoencoder:
            rconv3_dim = 64
            if self.handle_distort:
                rconv3_dim+=1
            self.recons_conv3 = nn.Conv2d(rconv3_dim, 3, 1, 1)

        self.upconv2 = upconv(up2_dim, 32, 3, 2, use_wrap_padding=use_wrap_padding)
        self.iconv2 = conv(iconv2_dim, 32, 3, 1, use_wrap_padding=use_wrap_padding)
        if self.autoencoder:
            self.recons_conv2 = nn.Conv2d(outconv_dim, 3, 1, 1)

        # fine-level conv

        self.out_conv = nn.Conv2d(outconv_dim, out_dim, 1, 1)
        if self.autoencoder:
            #1/2
            upconv1_dim=32
            rconv1_dim = inplanes
            iconv1_dim = inplanes*2
            if self.handle_distort:
                upconv1_dim+=1
                rconv1_dim+=1
                iconv1_dim+=1
            
            self.upconv1 = upconv(upconv1_dim, inplanes, 3, 2, use_wrap_padding=use_wrap_padding)
            self.iconv1 = conv(iconv1_dim, inplanes, 3, 1, use_wrap_padding=use_wrap_padding)
            self.recons_conv1 = nn.Conv2d(rconv1_dim, 3, 1, 1)
        
            # orig
            self.upconv0 = upconv(rconv1_dim, inplanes, 3, 2, use_wrap_padding=use_wrap_padding)
            self.recons_conv0 = nn.Conv2d(rconv1_dim, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_wrap_padding=False, handle_distort=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.handle_distort:
                tmp_inplanes = self.inplanes + 1
            else:
                tmp_inplanes = self.inplanes
            downsample = nn.Sequential(
                conv1x1(tmp_inplanes, planes * block.expansion, stride, use_wrap_padding=use_wrap_padding),
                norm_layer(planes * block.expansion, track_running_stats=False, affine=True)
            )
        # print("downsample:", downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, use_wrap_padding=use_wrap_padding, handle_distort=handle_distort))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, use_wrap_padding=use_wrap_padding, handle_distort=handle_distort))

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

    def expand_sin(self, x):
        if self.handle_distort:
            bs, _, height, width = x.shape
            sin_phi = torch.arange(0, height, dtype=torch.float32).cuda()
            sin_phi = torch.sin((sin_phi + 0.5) * math.pi / (height))
            sin_phi = sin_phi.view(1, 1, height, 1).expand(bs, 1, height, width)
            x = torch.cat([x, sin_phi],dim=1)  
        return x      


    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        # if self.cfg["handle_distort"]:
        #     bs, _, height, width = x.shape
        #     x = torch.cat([x, self.sin_phi.expand(bs, 1, height, width )], dim=1)
        if self.cfg["handle_distort"]:
            x = self.expand_sin(x)
        x0 = self.relu(self.bn1(self.conv1(x)))

        # x1 = self.layer1(self.expand_sin(x0))
        # x2 = self.layer2(self.expand_sin(x1))
        # x3 = self.layer3(self.expand_sin(x2))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1) #in BasicBlock, sin_phi has been appended
        x3 = self.layer3(x2)

        x = self.upconv3(self.expand_sin(x3))
        x = self.skipconnect(x2, x)
        x = self.iconv3(self.expand_sin(x))
        if self.autoencoder:
            outputs = {}
            outputs[("pred_img", 3)] = self.sigmoid(self.recons_conv3(self.expand_sin(x)))

        x = self.upconv2(self.expand_sin(x))
        x = self.skipconnect(x1, x)
        x = self.iconv2(self.expand_sin(x))
        x_out = self.out_conv(self.expand_sin(x))
        if self.autoencoder:
            outputs[("pred_img", 2)] = self.sigmoid(self.recons_conv2(self.expand_sin(x)))

            #1/2
            x = self.upconv1(self.expand_sin(x))
            # print("x.shape:", x.shape)
            
            x = self.skipconnect(x0, x)#orig
            if self.cfg["level"] == -2: #
                x_out = x.clone()
            
            x = self.iconv1(self.expand_sin(x))
            outputs[("pred_img", 1)] = self.sigmoid(self.recons_conv1(self.expand_sin(x)))

            #original res
            x = self.upconv0(self.expand_sin(x))
            # print("x.shape:", x.shape)
            # img_out = self.sigmoid(self.recons_conv(equi_x))
            if self.cfg["level"] == -1: #[-1, 0]
                x_out = x.clone()
            outputs[("pred_img", 0)] = self.sigmoid(self.recons_conv0(self.expand_sin(x)))

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

# class ResEncoder(nn.Module):
#     def __init__(self):
#         super(ResEncoder, self).__init__()
#         self.inplanes = 32
#         filters = [32, 64, 128]
#         layers = [2, 2, 2, 2]
#         out_planes = 32

#         norm_layer = nn.InstanceNorm2d
#         self._norm_layer = norm_layer
#         self.dilation = 1
#         block = BasicBlock
#         replace_stride_with_dilation = [False, False, False]
#         self.groups = 1
#         self.base_width = 64

#         self.conv1 = nn.Conv2d(12, self.inplanes, kernel_size=8, stride=2, padding=2,
#                                bias=False, padding_mode='zeros')
#         self.bn1 = norm_layer(self.inplanes, track_running_stats=False, affine=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self._make_layer(block, filters[0], layers[0], stride=2)
#         self.layer2 = self._make_layer(block, filters[1], layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, filters[2], layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])

#         # decoder
#         self.upconv3 = upconv(filters[2], filters[1], 3, 2)
#         self.iconv3 = conv(filters[1]*2, filters[1], 3, 1)
#         self.upconv2 = upconv(filters[1], filters[0], 3, 2)
#         self.iconv2 = conv(filters[0]*2, out_planes, 3, 1)
#         self.out_conv = nn.Conv2d(out_planes, out_planes, 1, 1)

#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion, track_running_stats=False, affine=True),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, 1, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))

#         return nn.Sequential(*layers)

#     def skipconnect(self, x1, x2):
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2))

#         # for padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

#         x = torch.cat([x2, x1], dim=1)
#         return x

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))

#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)

#         x = self.upconv3(x3)
#         x = self.skipconnect(x2, x)
#         x = self.iconv3(x)

#         x = self.upconv2(x)
#         x = self.skipconnect(x1, x)
#         x = self.iconv2(x)

#         x_out = self.out_conv(x)
#         return x_out

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
