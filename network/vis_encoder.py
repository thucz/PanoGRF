import torch.nn as nn
import torch

from network.ops import conv3x3, ResidualBlock, conv1x1

class DefaultVisEncoder(nn.Module):
    default_cfg={}
    def __init__(self, cfg):
        super().__init__()
        self.cfg={**self.default_cfg,**cfg}
        use_wrap_padding=self.cfg["use_wrap_padding"]
        norm_layer = lambda dim: nn.InstanceNorm2d(dim,track_running_stats=False,affine=True)
        if "level" in self.cfg and self.cfg["level"] in [-1]:#, -2]:
            in_dim=16+32
        else:
            in_dim=32+32
        self.out_conv=nn.Sequential(
            conv3x3(in_dim, 32, use_wrap_padding=use_wrap_padding),
            ResidualBlock(32, 32, norm_layer=norm_layer, use_wrap_padding=use_wrap_padding),
            ResidualBlock(32, 32, norm_layer=norm_layer, use_wrap_padding=use_wrap_padding),
            conv1x1(32, 32, use_wrap_padding=use_wrap_padding),
        )

    def forward(self, ray_feats, imgs_feats):
        # print("imgs_feats, ray_feats.shape:", imgs_feats.shape, ray_feats.shape)
        # import ipdb;ipdb.set_trace()
        # torch.cat([])
        if imgs_feats.shape[2:4] != ray_feats.shape[2:4]:
            i_feats = torch.nn.functional.interpolate(imgs_feats, (ray_feats.shape[2], ray_feats.shape[3]), mode='bilinear')
        else:
            i_feats = imgs_feats
        feats = self.out_conv(torch.cat([i_feats, ray_feats],1))
        return feats

name2vis_encoder={
    'default': DefaultVisEncoder,
}