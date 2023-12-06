import torch
import torch.nn as nn

from network.ops import interpolate_feats
from network.uncert_loss import compute_nll_loss, compute_perpoint_loss
from network.urf_loss import compute_urf_loss
import torch.nn.functional as F
import numpy as np
import cv2
import os
from .ae_loss import AELoss
class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys=keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass

class ConsistencyLoss(Loss):
    default_cfg={
        'use_ray_mask': False,
        'use_dr_loss': False,
        'use_dr_fine_loss': False,
        'use_nr_fine_loss': False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_prob','loss_prob_fine'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        
        if 'hit_prob_self' not in data_pr: return {}
        prob0 = data_pr['hit_prob_nr'].detach()     # qn,rn,dn
        prob1 = data_pr['hit_prob_self']            # qn,rn,dn
        # import ipdb;ipdb.set_trace()
        if self.cfg['use_ray_mask']:
            ray_mask = data_pr['ray_mask'].float()  # 1,rn
        else:
            ray_mask = 1
        ce = - prob0 * torch.log(prob1 + 1e-5) - (1 - prob0) * torch.log(1 - prob1 + 1e-5)

        outputs={'loss_prob': torch.mean(torch.mean(ce,-1),1)}
        
        if 'hit_prob_nr_fine' in data_pr:
            prob0 = data_pr['hit_prob_nr_fine'].detach()     # qn,rn,dn
            prob1 = data_pr['hit_prob_self_fine']            # qn,rn,dn
            ce = - prob0 * torch.log(prob1 + 1e-5) - (1 - prob0) * torch.log(1 - prob1 + 1e-5)
            outputs['loss_prob_fine']=torch.mean(torch.mean(ce,-1),1)
        return outputs

class RenderLoss(Loss):
    default_cfg={
        'use_ray_mask': True,
        'use_dr_loss': False,
        'use_dr_fine_loss': False,
        'use_nr_fine_loss': False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_rgb'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if self.cfg["fix_all"]:
            return {'loss_depth': torch.zeros([1], dtype=torch.float32, device=data_pr['pixel_colors_nr'].device)}
        # import ipdb;ipdb.set_trace()
        if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
            # import ipdb;ipdb.set_trace()
            rgb_gt = data_pr['pixel_colors_gt_fine'] # 1,rn,3
            # rgb_nr = data_pr['pixel_colors_nr_fine'] # 1,rn,3
            ray_mask = data_pr['ray_mask_fine'].float() # 1, rn
        else:
            rgb_gt = data_pr['pixel_colors_gt'] # 1,rn,3
            rgb_nr = data_pr['pixel_colors_nr'] # 1,rn,3
            ray_mask = data_pr['ray_mask'] # 1, rn
        
        

        # if "woblack" in self.cfg and self.cfg["woblack"]:
        #     rgb_gt = data_pr['pixel_colors_gt'] # 1,rn,3
        #     black_eps = 1e-3
        #     # import ipdb;ipdb.set_trace()
        #     bk_mask = torch.logical_and( torch.logical_and(rgb_gt[..., 0] < black_eps, rgb_gt[..., 1]< black_eps), rgb_gt[..., 2]< black_eps)
        #     bk_valid_mask = torch.logical_not( bk_mask)
        # else:
        # # debug = True
        # # if debug:
        bk_valid_mask = None
            

        if self.cfg["use_polar_weighted_loss"]:
            # import ipdb;ipdb.set_trace()
            weights_gt = data_pr['polar_weights'] # 1,rn,1

        def compute_loss(rgb_pr,rgb_gt):
            if bk_valid_mask is not None:
                if self.cfg["use_polar_weighted_loss"]:
                    loss=torch.sum((rgb_pr-rgb_gt)**2 * weights_gt,-1) #1, rn, 3, 1, rn, 1
                else:
                    loss=torch.sum((rgb_pr-rgb_gt)**2,-1)        # b,n            
                if self.cfg["use_polar_weighted_loss"]:
                    if self.cfg['use_ray_mask']:
                        # import ipdb;ipdb.set_trace()
                        # ray_mask = ray_mask.float() # 1,rn 
                        # loss = torch.sum(loss*ray_mask,1)/(torch.sum(ray_mask,1)+1e-3)
                        loss = torch.sum(loss*ray_mask * bk_valid_mask, dim=1)/(torch.sum(ray_mask * weights_gt.squeeze(-1) * bk_valid_mask, dim=1) + 1e-7)#1, rn->1, 
                    else:
                        loss = torch.sum(loss*bk_valid_mask, dim=1)/(torch.sum(weights_gt*bk_valid_mask.squeeze(-1), dim=1) + 1e-7) #1, rn->1, 
                        # loss = torch.mean(loss, 1)
                else:    
                    if self.cfg['use_ray_mask']:
                        # import ipdb;ipdb.set_trace()
                        # ray_mask = data_pr['ray_mask'].float() # 1,rn 
                        loss = torch.sum(loss*ray_mask*bk_valid_mask,1)/(torch.sum(ray_mask*bk_valid_mask,1)+1e-7)                
                    else:
                        loss = torch.sum(loss*bk_valid_mask, 1)/(torch.sum(bk_valid_mask, dim=1)+1e-7)
            else:
                if self.cfg["use_polar_weighted_loss"]:
                    loss=torch.sum((rgb_pr-rgb_gt)**2 * weights_gt,-1) #1, rn, 3, 1, rn, 1
                else:
                    loss=torch.sum((rgb_pr-rgb_gt)**2,-1)        # b,n            

                if self.cfg["use_polar_weighted_loss"]:
                    if self.cfg['use_ray_mask']:
                        # import ipdb;ipdb.set_trace()
                        # ray_mask = data_pr['ray_mask'].float() # 1,rn 
                        # loss = torch.sum(loss*ray_mask,1)/(torch.sum(ray_mask,1)+1e-3)
                        loss = torch.sum(loss*ray_mask, dim=1)/(torch.sum(ray_mask * weights_gt.squeeze(-1), dim=1) + 1e-7) #1, rn->1, 
                    else:
                        loss = torch.sum(loss, dim=1)/(torch.sum(weights_gt.squeeze(-1), dim=1)+1e-7) #1, rn->1, 
                        # loss = torch.mean(loss, 1)
                else:    
                    if self.cfg['use_ray_mask']:
                        # import ipdb;ipdb.set_trace()
                        # ray_mask = data_pr['ray_mask'].float() # 1,rn 
                        loss = torch.sum(loss*ray_mask,1)/(torch.sum(ray_mask,1)+1e-7)                
                    else:
                        loss = torch.mean(loss, 1)
            return loss
        

        if self.cfg["fix_coarse"]:
            results = {}
            rgb_nr = rgb_nr.detach()
        else:
            if ("one_mlp" in self.cfg and self.cfg["one_mlp"]):
                results = {}
            else:
                results = {'loss_rgb_nr': compute_loss(rgb_nr, rgb_gt)}
                if self.cfg['use_dr_loss']:
                    rgb_dr = data_pr['pixel_colors_dr']  # 1,rn,3
                    results['loss_rgb_dr'] = compute_loss(rgb_dr, rgb_gt)

        if self.cfg["use_hierarchical_sampling"] or ("diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"]):
            if self.cfg['use_dr_fine_loss']:
                results['loss_rgb_dr_fine'] = compute_loss(data_pr['pixel_colors_dr_fine'], rgb_gt)
            
            if self.cfg['use_nr_fine_loss']:
                results['loss_rgb_nr_fine'] = compute_loss(data_pr['pixel_colors_nr_fine'], rgb_gt)
        
        return results

class DepthLoss(Loss):
    default_cfg={
        'depth_correct_thresh': 0.02,
        'depth_loss_type': 'l2',
        'depth_loss_l1_beta': 0.05,
    }
    def __init__(self, cfg):
        super().__init__(['loss_depth'])
        self.cfg={**self.default_cfg,**cfg}
        if self.cfg['depth_loss_type']=='smooth_l1':
            self.loss_op=nn.SmoothL1Loss(reduction='none',beta=self.cfg['depth_loss_l1_beta'])

    def __call__(self, data_pr, data_gt, step, **kwargs):

        if 'true_depth' not in data_gt['ref_imgs_info']:
            if "diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"]:
                # import ipdb;ipdb.set_trace()
                return {'loss_depth': torch.zeros([1], dtype=torch.float32, device=data_pr['pixel_colors_nr_fine'].device)}
            else:
                return {'loss_depth': torch.zeros([1], dtype=torch.float32, device=data_pr['pixel_colors_nr'].device)}
        
        coords = data_pr['depth_coords'] # rfn,pn,2
        depth_pr = data_pr['depth_mean'] # rfn,pn
        
        depth_maps = data_gt['ref_imgs_info']['true_depth'] # rfn,1,h,w
        
        rfn, _, h, w = depth_maps.shape

        depth_gt = interpolate_feats(
            depth_maps,coords,h,w,padding_mode='border',align_corners=True)[...,0]   # rfn,pn

        # transform to inverse depth coordinate
        depth_range = data_gt['ref_imgs_info']['depth_range'] # rfn,2
        near, far = -1/depth_range[:,0:1], -1/depth_range[:,1:2] # rfn,1
        def process(depth):
            depth = torch.clamp(depth, min=1e-5)
            depth = -1 / depth
            depth = (depth - near) / (far - near)
            depth = torch.clamp(depth, min=0, max=1.0)
            return depth

        depth_gt = process(depth_gt)

        # compute loss
        def compute_loss(depth_pr):
            if self.cfg['depth_loss_type']=='l2':
                loss = (depth_gt - depth_pr)**2
            elif self.cfg['depth_loss_type']=='smooth_l1':
                loss = self.loss_op(depth_gt, depth_pr)

            # if data_gt['scene_name'].startswith('gso'):
            #     depth_maps_noise = data_gt['ref_imgs_info']['depth']  # rfn,1,h,w
            #     depth_aug = interpolate_feats(depth_maps_noise, coords, h, w, padding_mode='border', align_corners=True)[..., 0]  # rfn,pn
            #     depth_aug = process(depth_aug)
            #     mask = (torch.abs(depth_aug-depth_gt)<self.cfg['depth_correct_thresh']).float()
            #     loss = torch.sum(loss * mask, 1) / (torch.sum(mask, 1) + 1e-4)
            # else:
            loss = torch.mean(loss, 1)
            return loss

        outputs = {'loss_depth': compute_loss(depth_pr)}
        if 'depth_mean_fine' in data_pr:
            outputs['loss_depth_fine'] = compute_loss(data_pr['depth_mean_fine'])
        return outputs


class DepthFTLoss(Loss):
    default_cfg={
        'depth_correct_thresh': 0.02,
        'depth_loss_type': 'l2',
        'depth_loss_l1_beta': 0.05,
        'only_fine': False,
    }
    def __init__(self, cfg):
        super().__init__(['loss_depth_ft'])
        self.cfg={**self.default_cfg,**cfg}
        if self.cfg['depth_loss_type']=='smooth_l1':
            self.loss_op=nn.SmoothL1Loss(reduction='none',beta=self.cfg['depth_loss_l1_beta'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if self.cfg["fix_all"]:
            return {'loss_depth': torch.zeros([1], dtype=torch.float32, device=data_pr['pixel_colors_nr'].device)}
        if 'render_depth' not in self.cfg:
            return {'loss_depth': torch.zeros([1], dtype=torch.float32, device=data_pr['pixel_colors_nr'].device)}
        
        if "woblack" in self.cfg and self.cfg["woblack"]:
            rgb_gt = data_pr['pixel_colors_gt'] # 1,rn,3
            black_eps = 1e-3
            # import ipdb;ipdb.set_trace()
            bk_mask = torch.logical_and( torch.logical_and(rgb_gt[..., 0] < black_eps, rgb_gt[..., 1]< black_eps), rgb_gt[..., 2]< black_eps)
            bk_valid_mask = torch.logical_not( bk_mask)
        else:
            bk_valid_mask = None
        
        # coords = data_pr['depth_coords'] # rfn,pn,2
        if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
            pass
        else:
            depth_pr = data_pr['render_depth'] # rfn,pn
            if self.cfg["fix_coarse"]:
                depth_pr = depth_pr.detach()

            if 'render_uncert' in self.cfg and self.cfg["render_uncert"]:
                var_pr = data_pr['render_uncert']
                if self.cfg["fix_coarse"]:
                    var_pr = var_pr.detach()


            if 'perpoint_loss' in self.cfg and self.cfg['perpoint_loss']:
                dvals_pr = data_pr["render_dvals"]
                weights_pr = data_pr["render_weights"] 
            
            if 'urf_loss' in self.cfg and self.cfg["urf_loss"]:
                dvals_pr = data_pr["render_dvals"]
                weights_pr = data_pr["render_weights"]
        
        # depth_maps = data_gt['ref_imgs_info']['true_depth'] # rfn,1,h,w
        # import ipdb;ipdb.set_trace()
        que_imgs_info = data_pr['que_imgs_info']
        coords = que_imgs_info['coords']
        depth_maps = que_imgs_info['mvs_depth'].cuda()
        #  = depth_maps.shape[2:4]
        H, W = data_pr["ref_imgs_info"]["imgs"].shape[2:4]
        if data_pr["ref_imgs_info"]["imgs"].shape[2:4]!=depth_maps.shape[2:4]:
            depth_maps = F.interpolate(depth_maps, (H, W), mode='bilinear')

        que_id = data_pr["que_id"]
        os.makedirs("./vis_ft/"+self.cfg["name"], exist_ok=True)
        if 'render_uncert' in self.cfg and self.cfg["render_uncert"]:
            target_var = que_imgs_info['mvs_uncert'].cuda()
            var_map=target_var[0, 0].data.cpu().numpy()
            # import ipdb;ipdb.set_trace()
            import numpy as np
            import cv2
            var_norm = np.clip(var_map / 4, a_min=0, a_max=1.0) 
            var_norm = np.uint8(var_norm*255)
            vis_var = cv2.applyColorMap(var_norm, cv2.COLORMAP_JET)
            cv2.imwrite("vis_ft/"+self.cfg["name"]+"/var_rgb_"+str(que_id)+".jpg", vis_var)

        
        #visualize depth
        # import ipdb;ipdb.set_trace()
        dmap = depth_maps[0, 0].data.cpu().numpy()
        if self.cfg["has_depth"]:
            gt_depth = data_pr["all_info"]["depth"][que_id, 0].data.cpu().numpy()
            d_min = min(dmap.min(), gt_depth.min())
            d_max = max(dmap.max(), gt_depth.max())        
        else:
            d_min = dmap.min()
            d_max = dmap.max()

        d_norm = np.uint8((dmap-d_min)/(d_max-d_min)*255)
        d_rgb = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        
        if self.cfg["has_depth"]:
        
            gt_d_norm = np.uint8((gt_depth-d_min)/(d_max - d_min)*255)
            gt_d_rgb = cv2.applyColorMap(gt_d_norm, cv2.COLORMAP_JET)
        
        
        cv2.imwrite("vis_ft/"+self.cfg["name"]+"/d_rgb_"+str(que_id)+".jpg", d_rgb)
        if self.cfg["has_depth"]:
        
            cv2.imwrite("vis_ft/"+self.cfg["name"]+"/gt_d_rgb_"+str(que_id)+".jpg", gt_d_rgb)

        color = np.uint8(data_pr["all_info"]["imgs"][que_id].permute((1, 2, 0)).data.cpu().numpy()*255)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        cv2.imwrite("vis_ft/"+self.cfg["name"]+"/color_"+str(que_id)+".jpg", color)
        if self.cfg["has_depth"]:
            # import ipdb;ipdb.set_trace()
            depth_gt_mask = gt_depth>=self.cfg["min_depth"]
            error_l2 = (((gt_depth - dmap)** 2) * depth_gt_mask).mean()
            with open("vis_ft/"+self.cfg["name"]+"/error_l2_"+str(que_id)+".txt", "w") as fp:
                # import ipdb;ipdb.set_trace()
                fp.write("error_l2:"+str(error_l2))

        # cv2.imwrite("")
        # import ipdb;ipdb.set_trace()
        qn, _, h, w = depth_maps.shape
        depth_mvs = interpolate_feats( #qn, pn
            depth_maps,coords,h,w,padding_mode='border',align_corners=False)[...,0]   # qn,pn
        # import ipdb;ipdb.set_trace()
        if self.cfg["render_uncert"]:
            var_mvs = interpolate_feats(
                target_var, coords, h,w,padding_mode='border',align_corners=False)[...,0]
            # import ipdb;ipdb.set_trace()
            if "fix_uncert" in self.cfg and self.cfg["fix_uncert"]:
                var_mvs = torch.ones_like(var_mvs)*self.cfg["fix_uncert"] # 0.01**2
            if "clamp_uncert" in self.cfg and self.cfg["clamp_uncert"]:
                var_mvs = torch.clamp(var_mvs, max=self.cfg["max_uncert"])



        if "use_polar_weighted_loss" in self.cfg and self.cfg["use_polar_weighted_loss"]:
            # import ipdb;ipdb.set_trace()            
            weights_gt = data_pr['polar_weights'].squeeze(2) # 1,rn,1
        else:
            weights_gt = None
    
        # transform to inverse depth coordinate
        depth_range = data_pr['ref_imgs_info']['depth_range'] # rfn,2
        near, far = -1/depth_range[:,0:1], -1/depth_range[:,1:2] # rfn,1
        
        def process(depth):
            # depth = torch.clamp(depth, min=1e-5)
            # mask = depth >= self.cfg["min_depth"]
            depth = torch.clamp(depth, min=self.cfg["min_depth"], max=self.cfg["max_depth"])
            depth = -1 / depth
            depth = (depth - near) / (far - near)
            depth = torch.clamp(depth, min=0, max=1.0)
            return depth #, mask
        # mask = depth_mvs >= self.cfg["min_depth"]
        # if self.cfg["depth_norm"]:
        #     depth_mvs = process(depth_mvs)
        #     depth_pr = process(depth_pr)
        # else:
            # mask = np.ones_like(depth_pr)
        # compute loss
        def compute_loss(depth_pr):
            # import ipdb;ipdb.set_trace()

            if self.cfg['depth_loss_type']=='l2':
                loss = (depth_mvs - depth_pr)**2 #* mask
            elif self.cfg['depth_loss_type']=='smooth_l1':
                loss = self.loss_op(depth_mvs, depth_pr) #* mask

            # if data_gt['scene_name'].startswith('gso'):
            #     depth_maps_noise = data_gt['ref_imgs_info']['depth']  # rfn,1,h,w
            #     depth_aug = interpolate_feats(depth_maps_noise, coords, h, w, padding_mode='border', align_corners=True)[..., 0]  # rfn,pn
            #     depth_aug = process(depth_aug)
            #     mask = (torch.abs(depth_aug-depth_gt)<self.cfg['depth_correct_thresh']).float()
            #     loss = torch.sum(loss * mask, 1) / (torch.sum(mask, 1) + 1e-4)
            # else:
            if weights_gt is not None:
                loss = torch.sum(loss*weights_gt, dim=1)/torch.sum(weights_gt, dim=1)
            else:
                loss = torch.mean(loss, 1)
            # import ipdb;ipdb.set_trace()

            return loss
        if "urf_loss" in self.cfg and self.cfg["urf_loss"]:
            # depth_pr, var_pr, depth_mvs, var_mvs
            if self.cfg["fix_coarse"]:
                outputs = {}
            else:

                urf_loss = compute_urf_loss(self.cfg, depth_pr, dvals_pr, weights_pr, depth_mvs, var_mvs)
                # if self.cfg["use_hi"]
                outputs = {'loss_depth_ft_urf': self.cfg["urf_loss_lambda"] * urf_loss}            
            if self.cfg["use_hierarchical_sampling"]:
                urf_loss_fine = compute_urf_loss(self.cfg, data_pr['render_depth_fine'], data_pr["render_dvals_fine"], data_pr["render_weights_fine"], depth_mvs, var_mvs)
                outputs["loss_depth_ft_urf_fine"] = self.cfg["urf_loss_lambda"]*urf_loss_fine
            
        elif "perpoint_loss" in self.cfg and self.cfg["perpoint_loss"]:
            if self.cfg["fix_coarse"]:
                weights_pr = weights_pr.detach()

                outputs = {}
            else:
                perpoint_loss = compute_perpoint_loss(self.cfg, dvals_pr, weights_pr, depth_mvs, weights_gt)

                outputs = {'loss_perpoint': self.cfg["perpoint_loss_lambda"] * perpoint_loss}
            if self.cfg["use_hierarchical_sampling"]:
                # data_pr['render_depth_fine'], data_pr['render_uncert_fine'], depth_mvs, var_mvs
                perpoint_loss_fine = compute_perpoint_loss(self.cfg, data_pr["render_dvals_fine"], data_pr["render_weights_fine"], depth_mvs, weights_gt)
        
            outputs["loss_perpoint_fine"] = self.cfg["perpoint_loss_lambda"] * perpoint_loss_fine
        elif self.cfg["render_uncert"]:
            if "coarse_clip_sigma" in self.cfg:
                coarse_clip_sigma = self.cfg["coarse_clip_sigma"]
            else:
                coarse_clip_sigma = 0.0
            if "fine_clip_sigma" in self.cfg:
                fine_clip_sigma = self.cfg["fine_clip_sigma"]
            else:
                fine_clip_sigma = 0.0

            if self.cfg["only_fine"]:
                outputs = {}
                # pass
            else:
                if self.cfg["fix_coarse"]:
                    outputs = {}
                else:
                    # import ipdb;ipdb.set_trace()
                    if "one_mlp" in self.cfg and self.cfg["one_mlp"] and ("diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"]):
                        outputs = {}
                    else:
                        if "coarse_use_mse" in self.cfg and self.cfg["coarse_use_mse"]:
                            outputs = {'loss_depth_ft_coarse_mse': self.cfg["depth_ft_lambda"] * compute_loss(depth_pr)}    
                        else:
                            outputs = {'loss_depth_ft_uncert': self.cfg["depth_ft_uncert_lambda"] * compute_nll_loss(self.cfg, depth_pr, var_pr, depth_mvs, var_mvs, coarse_clip_sigma, weights_gt, bk_valid_mask)}
                
            # if 'depth_mean_fine' in data_pr:
                # outputs['loss_depth_fine'] = compute_loss(data_pr['depth_mean_fine'])
            # import ipdb;ipdb.set_trace()
            if self.cfg["use_hierarchical_sampling"] or ("diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"]):
                if "fine_use_mse" in self.cfg and self.cfg["fine_use_mse"]:
                    outputs['loss_depth_ft_mse_fine'] = self.cfg["depth_ft_lambda"] * compute_loss(data_pr['render_depth_fine'])
                else:
                    outputs['loss_depth_ft_uncert_fine'] = self.cfg["depth_ft_uncert_lambda"] * compute_nll_loss(self.cfg, data_pr['render_depth_fine'], data_pr['render_uncert_fine'], depth_mvs, var_mvs, fine_clip_sigma, weights_gt, bk_valid_mask)
                if "add_mse_loss" in self.cfg and self.cfg["add_mse_loss"]:
                    outputs['loss_depth_ft_mse_fine'] = self.cfg["depth_ft_lambda"] * compute_loss(data_pr['render_depth_fine'])

        else:
            if self.cfg["fix_coarse"]:
                outputs = {}
            else:
                outputs = {'loss_depth_ft': self.cfg["depth_ft_lambda"] * compute_loss(depth_pr)}
            # import ipdb;ipdb.set_trace()

            # if 'depth_mean_fine' in data_pr:
                # outputs['loss_depth_fine'] = compute_loss(data_pr['depth_mean_fine'])
            if self.cfg["use_hierarchical_sampling"] or ("diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"]):
                outputs['loss_depth_ft_fine'] = self.cfg["depth_ft_lambda"] * compute_loss(data_pr['render_depth_fine'])
        return outputs

name2loss={
    'render': RenderLoss,
    'depth': DepthLoss,
    'consist': ConsistencyLoss,
    'depth_ft': DepthFTLoss,
    'ae_loss': AELoss
}