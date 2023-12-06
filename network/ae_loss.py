from __future__ import absolute_import, division, print_function
from pathlib import Path
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
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

class AELoss():
    default_cfg={
        # 'use_ray_mask': False,
        # 'use_dr_loss': False,
        # 'use_dr_fine_loss': False,
        # 'use_nr_fine_loss': False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        # super().__init__([f'loss_prob','loss_prob_fine'])
        self.ssim = SSIM()
        self.criterion = nn.SmoothL1Loss(reduction='none')


    def __call__(self, data_pr, data_gt, step, **kwargs):
        loss_dict = {}
        # target = data_gt?
        # import ipdb;ipdb.set_trace()

        ref_imgs_info = data_gt["ref_imgs_info"]
        target = ref_imgs_info["imgs"]#?
        outputs = data_pr["ae_outputs"]
        model_name = self.cfg['name']
        scales = 4
        for scale in range(scales):
            """
            initialization
            """
            pred = outputs[("pred_img", scale)]
            bs,_,h,w = pred.size()
            target = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)
            min_reconstruct_loss = self.compute_reprojection_loss(pred, target)
            # import ipdb;ipdb.set_trace()
            loss_dict[('min_reconstruct_loss', scale)] = min_reconstruct_loss.mean()/scales
            if step % 500 == 0:
                Path(f'data/ae_recons/{model_name}').mkdir(exist_ok=True, parents=True)
                img_path = os.path.join(f'data/ae_recons/{model_name}', 'auto_{:0>4d}_{}.png'.format(step // 2000, scale))
                plt.imsave(img_path, pred[0].transpose(0,1).transpose(1,2).data.cpu().numpy())
                img_path = os.path.join(f'data/ae_recons/{model_name}', 'img_{:0>4d}_{}.png'.format(step // 2000, scale))
                plt.imsave(img_path, target[0].transpose(0, 1).transpose(1, 2).data.cpu().numpy())
        # 
        # results = {}
        loss = sum(_value for _key, _value in loss_dict.items())
        results = {"loss_ae": self.cfg["lambda_ae"]*loss.unsqueeze(0)}
        return results

   
    def WSmoothL1(self, loss, x):
        bs, _, height, width = x.shape
        sin_phi = torch.arange(0, height, dtype=torch.float32).cuda()
        sin_phi = torch.sin((sin_phi + 0.5) * math.pi / (height))
        sin_phi = sin_phi.view(1, 1, height, 1).expand(bs, 1, height, width)
        return loss * sin_phi
        

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_reprojection_loss(self, pred, target):
        # photometric_loss = self.robust_l1(pred, target).mean(1, True)
        if "ae_type" in self.cfg and self.cfg["ae_type"] == 1:
            photometric_loss = self.robust_l1(pred, target).mean(1, True)
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        else:
            reprojection_loss = self.criterion(pred, target).mean(1, True)
        if "ae_type" in self.cfg and self.cfg["ae_type"] == 1:
            pass
        else:
            if self.cfg["use_polar_weighted_loss"]:
                reprojection_loss = self.WSmoothL1(reprojection_loss, pred)
        # ssim_loss = self.ssim(pred, target).mean(1, True)
        # reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss
