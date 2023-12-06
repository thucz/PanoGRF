import time
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
from skimage.io import imsave
from tqdm import tqdm

from dataset.database import M3DDatabase, ReplicaDatabase
from network.aggregate_net import name2agg_net
from network.dist_decoder import name2dist_decoder
from network.init_net import name2init_net, CostVolumeInitNet
from network.ops import ResUNetLight
from network.resnet_erp_tp import ResUNetLight_ERP_TP
from network.sph_solver import SphericalHarmonicsSolver
from network.vis_encoder import name2vis_encoder
from network.render_ops import *
from utils.base_utils import to_cuda, load_cfg, color_map_backward, get_coords_mask
from utils.draw_utils import concat_images_list
from utils.imgs_info import build_imgs_info, imgs_info_to_torch, imgs_info_slice
# from utils.view_select import compute_nearest_camera_indices, select_working_views
import copy
from .spt_utils import Utils
from .ray_utils import get_sphere_ray_directions
from data_readers.habitat_data_neuray_ft import HabitatImageGeneratorFT
from data_readers.habitat_data_neuray_ft_lmdb import HabitatImageGeneratorFT_LMDB
from network.sample_utils import precompute_depth_sampling, precompute_quadratic_samples

from network.diner_depth_guided_sample import select_depth
from network.original_depth_guided_sample import sample_depthguided
# from network.depth2normal import depth2normal
from network.orig_diner_depth2normal import depth2normal

class NeuralRayBaseRenderer(nn.Module):
    base_cfg={
        'vis_encoder_type': 'default',
        'vis_encoder_cfg': {},

        'dist_decoder_type': 'mixture_logistics',
        'dist_decoder_cfg': {},
        'autoencoder': False,

        'agg_net_type': 'default',
        'agg_net_cfg': {},

        'use_hierarchical_sampling': False,
        'fine_agg_net_cfg': {},
        'fine_dist_decoder_cfg': {},
        'fine_depth_sample_num': 64,
        'fine_depth_use_all': False,
        'ray_batch_num': 2048,
        'depth_sample_num': 64,
        'alpha_value_ground_state': -15,
        'use_dr_prediction': False,
        'use_nr_color_for_dr': False,
        'use_self_hit_prob': False,
        'use_ray_mask': True,
        'ray_mask_view_num': 1,
        'ray_mask_point_num': 8,
        'patchsize': (128, 128),

        'render_depth': False,
    }
    def __init__(self,cfg):
        super().__init__()
        self.cfg = {**self.base_cfg, **cfg}
        if "sample_num" in cfg:
            self.cfg["agg_net_cfg"]["sample_num"] = self.cfg["sample_num"]
            self.cfg["fine_agg_net_cfg"]["sample_num"] = self.cfg["sample_num"]
        if "level" in cfg:
            self.cfg["agg_net_cfg"]["level"] = self.cfg["level"]
            self.cfg["fine_agg_net_cfg"]["level"] = self.cfg["level"]
        if "wo_geometry" in cfg:
            self.cfg["agg_net_cfg"]["wo_geometry"] = self.cfg["wo_geometry"]
            self.cfg["fine_agg_net_cfg"]["wo_geometry"] = self.cfg["wo_geometry"]
        if "wo_appearance" in cfg:
            self.cfg["agg_net_cfg"]["wo_appearance"] = self.cfg["wo_appearance"]
            self.cfg["fine_agg_net_cfg"]["wo_appearance"] = self.cfg["wo_appearance"]


        
            
            
        use_wrap_padding = self.cfg["use_wrap_padding"]
        self.spt_utils = Utils(copy.deepcopy(self.cfg))#todo: cfg
        self.directions = get_sphere_ray_directions(
            self.spt_utils)  # (h, w, 3)        
        

        #todo:(ERP+TP)
        self.vis_encoder = name2vis_encoder[self.cfg['vis_encoder_type']](self.cfg)#['vis_encoder_cfg'])
        #not revised(MLP)
        self.dist_decoder = name2dist_decoder[self.cfg['dist_decoder_type']](self.cfg['dist_decoder_cfg'])
        #todo:(ERP+TP)
        if "local_feature_type" in self.cfg and self.cfg["local_feature_type"] == "ERP+TP":
            # self.image_encoder = ResUNetLight_ERP_TP()
            npatches_dict = {3:10, 4:18, 5:26, 6:46}
            # import ipdb;ipdb.set_trace()
            self.image_encoder = ResUNetLight_ERP_TP(3, [1,2,6,4], 32, inplanes=16, use_wrap_padding=use_wrap_padding,               
                 fusion_type=self.cfg["fusion"], se_in_fusion=self.cfg["se_in_fusion"],  nrows=self.cfg['nrows'], 
                 npatches=npatches_dict[self.cfg["nrows"]], patch_size=self.cfg["patchsize"], fov=self.cfg["fov"], \
                 autoencoder=self.cfg["autoencoder"]
            )
        else:
            self.image_encoder = ResUNetLight(self.cfg, 3, [1,2,6,4], 32, inplanes=16, use_wrap_padding=use_wrap_padding,
                autoencoder=self.cfg["autoencoder"])

        #MLP
        self.agg_net = name2agg_net[self.cfg['agg_net_type']](self.cfg['agg_net_cfg'])

        if self.cfg['use_hierarchical_sampling']:
            if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
                pass
            else:
                self.fine_dist_decoder = name2dist_decoder[self.cfg['dist_decoder_type']](self.cfg['fine_dist_decoder_cfg'])
                self.fine_agg_net = name2agg_net[self.cfg['agg_net_type']](self.cfg['fine_agg_net_cfg'])
        # if self.cfg['use_dr_prediction'] and not self.cfg['use_nr_color_for_dr']:
        self.sph_fitter = SphericalHarmonicsSolver(3)
    def predict_proj_ray_prob(self, prj_dict, ref_imgs_info, que_dists, is_fine):
        rfn, qn, rn, dn, _ = prj_dict['pts'].shape
        # decode ray prob
        if is_fine:
            prj_mean, prj_var, prj_vis, prj_aw = self.fine_dist_decoder(prj_dict['ray_feats'])
        else:
            prj_mean, prj_var, prj_vis, prj_aw = self.dist_decoder(prj_dict['ray_feats'])

        alpha_values, visibility, hit_prob = self.dist_decoder.compute_prob(
            prj_dict['depth'].squeeze(-1),que_dists.unsqueeze(0),prj_mean,prj_var,
            prj_vis, prj_aw, True, ref_imgs_info['depth_range'])
        # post process
        prj_dict['alpha'] = alpha_values.reshape(rfn,qn,rn,dn,1) #* prj_dict['mask'] + \
                            # (1 - prj_dict['mask']) * self.cfg['alpha_value_ground_state']
        prj_dict['vis'] = visibility.reshape(rfn,qn,rn,dn,1) #* prj_dict['mask']
        prj_dict['hit_prob'] = hit_prob.reshape(rfn,qn,rn,dn,1) #* prj_dict['mask']
        return prj_dict

    def predict_alpha_values_dr(self, prj_dict):
        eps = 1e-5
        # predict alpha values for query ray
        prj_alpha, prj_vis = prj_dict['alpha'], prj_dict['vis']
        alpha = torch.sum(prj_vis * prj_alpha, 0) / (torch.sum(prj_vis, 0) + eps)  # qn,rn,dn,1
        # invalid_ray_mask = torch.sum(prj_dict['mask'].int().squeeze(-1), 0) == 0
        # alpha = alpha #* (1 - invalid_ray_mask.float().unsqueeze(-1)) + \
                #invalid_ray_mask.float().unsqueeze(-1) * self.cfg['alpha_value_ground_state']
        rfn, qn, rn, dn, _ = prj_alpha.shape
        return alpha.reshape(qn, rn, dn)

    def predict_colors_dr(self,prj_dict,que_dir):
        eps = 1e-3
        prj_hit_prob, prj_rgb, prj_dir = prj_dict['hit_prob'], prj_dict['rgb'], prj_dict['dir']
        rfn, qn, rn, dn, _ = prj_rgb.shape
        pn = qn * rn * dn
        que_dir = que_dir.reshape(pn, 3)  # pn,3
        prj_dir = prj_dir.reshape(rfn, pn, 3)
        prj_rgb = prj_rgb.reshape(rfn, pn, 3)
        prj_hit_prob = prj_hit_prob.reshape(rfn,pn,1)
        prj_weights = prj_hit_prob / (torch.sum(prj_hit_prob, 0, keepdim=True) + eps) # rfn,pn,3

        # pn,k,3
        theta = self.sph_fitter(prj_dir.permute(1,0,2),
                                prj_rgb.permute(1,0,2),
                                prj_weights.squeeze(-1).permute(1,0)) # pn,rfn
        colors = self.sph_fitter.predict(que_dir.unsqueeze(1),theta)
        colors = colors.squeeze(1).reshape(qn,rn,dn,3)
        return colors

    def direct_rendering(self, prj_dict, que_dir, colors_nr=None):
        alpha_values = self.predict_alpha_values_dr(prj_dict)               # qn,rn,dn
        if self.cfg['use_nr_color_for_dr']:
            colors = colors_nr
        else:
            colors = self.predict_colors_dr(prj_dict,que_dir)               # qn,rn,dn,3
        # the alpha values is *logits* now, we decode it to *real alpha values*
        alpha_values = self.dist_decoder.decode_alpha_value(alpha_values)   # qn,rn,dn
        hit_prob = alpha_values2hit_prob(alpha_values)                      # qn,rn,dn
        pixel_colors = torch.sum(hit_prob.unsqueeze(-1)*colors,2)
        return hit_prob, colors, pixel_colors

    def get_img_feats(self,ref_imgs_info, prj_dict):
        rfn, _, h, w = ref_imgs_info['imgs'].shape
        rfn, qn, rn, dn, _ = prj_dict['pts'].shape

        img_feats = ref_imgs_info['img_feats']
        prj_img_feats = interpolate_feature_map(img_feats, prj_dict['pts'].reshape(rfn, qn * rn * dn, 2),
                                                h, w,)
        prj_dict['img_feats'] = prj_img_feats.reshape(rfn, qn, rn, dn, -1)
        return prj_dict

    def predict_self_hit_prob_impl(self, que_ray_feats, que_depth, que_dists, depth_range, is_fine):
        if is_fine: ops = self.fine_dist_decoder
        else: ops = self.dist_decoder
        mean, var, vis, aw = ops(que_ray_feats)  # qn,rn,1
        if aw is not None: aw = aw.unsqueeze(2)
        if vis is not None: vis = vis.unsqueeze(2)
        if mean is not None: mean = mean.unsqueeze(2)
        if var is not None: var = var.unsqueeze(2)
        # qn, rn, dn
        _, _, hit_prob_que = ops.compute_prob(que_depth, que_dists, mean, var, vis, aw, False, depth_range)
        return hit_prob_que

    def predict_self_hit_prob(self, que_imgs_info, que_depth, que_dists, is_fine):
        _, _, h, w = que_imgs_info['imgs'].shape
        qn, rn, _ = que_imgs_info['coords'].shape
        # mask = torch.ones([qn, rn], dtype=torch.float32, device=que_imgs_info['coords'].device)
        que_ray_feats = interpolate_feature_map(que_imgs_info['ray_feats'], que_imgs_info['coords'], h, w)  # qn,rn,f
        hit_prob_que = self.predict_self_hit_prob_impl(que_ray_feats, que_depth, que_dists, que_imgs_info['depth_range'], is_fine)
        return hit_prob_que

    def network_rendering(self, prj_dict, que_dir, is_fine):
        if is_fine:
            density, colors = self.fine_agg_net(prj_dict, que_dir)
        else:
            density, colors = self.agg_net(prj_dict, que_dir)

        alpha_values = 1.0 - torch.exp(-torch.relu(density))
        hit_prob = alpha_values2hit_prob(alpha_values)
        pixel_colors = torch.sum(hit_prob.unsqueeze(-1)*colors,2)
        return hit_prob, colors, pixel_colors, density


    #todo?
    def render_by_depth(self, que_depth, que_imgs_info, ref_imgs_info, is_train, is_fine, is_perspec=False):

        ref_imgs_info = ref_imgs_info.copy()
        que_imgs_info = que_imgs_info.copy()
        if self.cfg["debug"]:
            pass
        else: 
            que_dists = depth2inv_dists(que_depth,que_imgs_info['depth_range'])
        if is_perspec:
            que_pts, que_dir = depth2points_perspec(que_imgs_info, que_depth)        
        else:
            que_pts, que_dir = depth2points_spherical(que_imgs_info, que_depth, self.spt_utils)        

        prj_dict = project_points_dict(ref_imgs_info, que_pts, self.spt_utils)
        # if 
        # import ipdb;ipdb.set_trace()

        if self.cfg["debug"]:
            pass
        else:
            prj_dict = self.predict_proj_ray_prob(prj_dict, ref_imgs_info, que_dists, is_fine)        
        prj_dict = self.get_img_feats(ref_imgs_info, prj_dict)
        if self.cfg["debug"]:
            outputs={}
            if 'imgs' in que_imgs_info:
                # if self.cfg["start_debug"]:
                #     import ipdb;ipdb.set_trace()
                outputs['pixel_colors_gt'] = interpolate_feats(
                    que_imgs_info['imgs'], que_imgs_info['coords'], align_corners=True)
                
            
            # import ipdb;ipdb.set_trace() #(rfn,qn,rn,dn,-1) , 2, 1, 2048, 1, -1
            outputs["pts"] = prj_dict["pts"].squeeze(3).squeeze(1)
            outputs["rgb"] = prj_dict["rgb"].squeeze(3).squeeze(1)
            outputs["depths"] = prj_dict["depth"].squeeze(3).squeeze(1)
            outputs["dir"] = prj_dict["dir"].squeeze(3).squeeze(1)

        else:

            #ray_feats, img_feats, alpha, vis, hit_prob in prj_dict produce nan?important
            hit_prob_nr, colors_nr, pixel_colors_nr, density_nr = self.network_rendering(prj_dict, que_dir, is_fine)
            if torch.isnan(hit_prob_nr).any():
                import ipdb;ipdb.set_trace()
            outputs={'pixel_colors_nr': pixel_colors_nr, 'hit_prob_nr': hit_prob_nr, "colors_nr": colors_nr, "density_nr": density_nr}

            # direct rendering
            # if self.cfg['use_dr_prediction']:
            #     hit_prob_dr, colors_dr, pixel_colors_dr = self.direct_rendering(prj_dict, que_dir, colors_nr)
            #     outputs['pixel_colors_dr'] = pixel_colors_dr
            #     outputs['hit_prob_dr'] = hit_prob_dr

            # predict query hit prob
            if is_train and self.cfg['use_self_hit_prob']:
                outputs['hit_prob_self'] = self.predict_self_hit_prob(que_imgs_info, que_depth, que_dists, is_fine)

            if 'imgs' in que_imgs_info:
                # if self.cfg["start_debug"]:
                #     import ipdb;ipdb.set_trace()
                outputs['pixel_colors_gt'] = interpolate_feats(
                    que_imgs_info['imgs'], que_imgs_info['coords'], align_corners=True)
                if self.cfg["use_polar_weighted_loss"]:
                    outputs['polar_weights'] = interpolate_feats(
                        que_imgs_info['polar_weights'], que_imgs_info['coords'], align_corners=True)
            elif 'cube_imgs' in que_imgs_info:
                outputs['pixel_colors_gt'] = interpolate_feats(
                    que_imgs_info['cube_imgs'], que_imgs_info['coords'], align_corners=True)
                

            if self.cfg['use_ray_mask']:
                # prj_dict['mask']
                # (rfn,qn,rn,dn,-1)#reference views, query views,
                rfn, qn, rn, dn, _ = prj_dict["rgb"].shape
                mask = torch.ones([rfn, qn, rn, dn, 1], dtype=torch.float32, device=prj_dict["rgb"].device)
                outputs['ray_mask'] = torch.sum(mask.int(),0)>=self.cfg['ray_mask_view_num'] # qn,rn,dn,1
                outputs['ray_mask'] = torch.sum(outputs['ray_mask'],2)>self.cfg['ray_mask_point_num'] # qn,rn
                outputs['ray_mask'] = outputs['ray_mask'][...,0]
            # if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
            #     pass
            # else:
            if self.cfg['render_depth']:
                # qn,rn,dn
                outputs['render_depth'] = torch.sum(hit_prob_nr * que_depth, -1) # qn,rn
            if self.cfg['render_uncert']:
                # outputs['']
                # que_depth = qn, rn
                # pred_mean  = outputs["render_depth"]
                # dist_vals = que_depth
                # weights = hit_prob_nr
                # import ipdb;ipdb.set_trace()
                outputs['render_uncert'] = ((que_depth - outputs["render_depth"].unsqueeze(-1)).pow(2) * hit_prob_nr).sum(-1) + 1e-5
                
            if "perpoint_loss" in self.cfg and self.cfg["perpoint_loss"]:
                outputs["render_weights"] = hit_prob_nr
                outputs["render_dvals"] = que_depth
        return outputs


    #todo?
    def diner_render_by_depth(self, diner_que_depth, que_imgs_info, ref_imgs_info, is_train, is_fine, is_perspec=False):
        ref_imgs_info = ref_imgs_info.copy()
        que_imgs_info = que_imgs_info.copy()
        que_dists = depth2inv_dists(diner_que_depth,que_imgs_info['depth_range'])        
        if is_perspec:
            que_pts, que_dir = depth2points_perspec(que_imgs_info, diner_que_depth)        
        else:
            que_pts, que_dir = depth2points_spherical(que_imgs_info, diner_que_depth, self.spt_utils)        

        # if torch.isnan(que_pts).any():
        #     import ipdb;ipdb.set_trace()
        if "backface_culling" in self.cfg:
            include_norm = self.cfg["backface_culling"]
        else:
            include_norm = False
        prj_depth_info_dict = project_points_dict_diner(ref_imgs_info, que_pts, self.spt_utils, include_norm = include_norm)
        # select_pts
        #todo: select points for later process. 
        # new_que_depth = select_depth(self.cfg, prj_depth_info_dict, diner_que_depth, que_dir, include_norm=include_norm)

        # import ipdb;ipdb.set_trace()
        new_que_depth = sample_depthguided(self.cfg, ref_imgs_info, prj_depth_info_dict, diner_que_depth, que_dir, n_samples=self.cfg['n_samples'], \
            n_candidates=self.cfg['n_candidates'], n_gaussian=self.cfg["n_gaussian"], depth_diff_max=0.05, include_norm=include_norm)

        
        if "contain_uniform" in self.cfg and self.cfg["contain_uniform"]:
            coarse_uniform_que_depth, _ = sample_depth(self.cfg, que_imgs_info['coords'], self.cfg["n_uniform"], False, use_disp="inv_uniform" in self.cfg and self.cfg["inv_uniform"])
            new_que_depth = torch.cat([new_que_depth, coarse_uniform_que_depth], dim=-1)
            new_que_depth, _ = new_que_depth.sort(dim=-1)
            # import ipdb;ipdb.set_trace()
        # import ipdb;ipdb.set_trace()    
        que_dists = depth2inv_dists(new_que_depth,que_imgs_info['depth_range'])
        if is_perspec:
            que_pts, que_dir = depth2points_perspec(que_imgs_info, new_que_depth)        
        else:
            que_pts, que_dir = depth2points_spherical(que_imgs_info, new_que_depth, self.spt_utils)        

        prj_dict = project_points_dict(ref_imgs_info, que_pts, self.spt_utils)
        # if 
        # import ipdb;ipdb.set_trace()

        if self.cfg["debug"]:
            pass
        else:
            prj_dict = self.predict_proj_ray_prob(prj_dict, ref_imgs_info, que_dists, is_fine)
        
        prj_dict = self.get_img_feats(ref_imgs_info, prj_dict)
        if self.cfg["debug"]:
            outputs={}
            if 'imgs' in que_imgs_info:
                # if self.cfg["start_debug"]:
                #     import ipdb;ipdb.set_trace()
                outputs['pixel_colors_gt'] = interpolate_feats(
                    que_imgs_info['imgs'], que_imgs_info['coords'], align_corners=True)
            # import ipdb;ipdb.set_trace() #(rfn,qn,rn,dn,-1) , 2, 1, 2048, 1, -1
            outputs["pts"] = prj_dict["pts"].squeeze(3).squeeze(1)
            outputs["rgb"] = prj_dict["rgb"].squeeze(3).squeeze(1)
            outputs["depths"] = prj_dict["depth"].squeeze(3).squeeze(1)
            outputs["dir"] = prj_dict["dir"].squeeze(3).squeeze(1)

        else:

            #ray_feats, img_feats, alpha, vis, hit_prob in prj_dict produce nan?important
            hit_prob_nr, colors_nr, pixel_colors_nr, density_nr = self.network_rendering(prj_dict, que_dir, is_fine)
            if torch.isnan(hit_prob_nr).any():
                import ipdb;ipdb.set_trace()
            outputs={'pixel_colors_nr': pixel_colors_nr, 'hit_prob_nr': hit_prob_nr, "colors_nr": colors_nr, "density_nr": density_nr}
            outputs["que_depth"] = new_que_depth
            # direct rendering
            # if self.cfg['use_dr_prediction']:
            #     hit_prob_dr, colors_dr, pixel_colors_dr = self.direct_rendering(prj_dict, que_dir, colors_nr)
            #     outputs['pixel_colors_dr'] = pixel_colors_dr
            #     outputs['hit_prob_dr'] = hit_prob_dr

            # # predict query hit prob
            if is_train and self.cfg['use_self_hit_prob']:
                outputs['hit_prob_self'] = self.predict_self_hit_prob(que_imgs_info, que_depth, que_dists, is_fine)

            if 'imgs' in que_imgs_info:
                # if self.cfg["start_debug"]:
                #     import ipdb;ipdb.set_trace()
                outputs['pixel_colors_gt'] = interpolate_feats(
                    que_imgs_info['imgs'], que_imgs_info['coords'], align_corners=True)
                if self.cfg["use_polar_weighted_loss"]:
                    outputs['polar_weights'] = interpolate_feats(
                        que_imgs_info['polar_weights'], que_imgs_info['coords'], align_corners=True)

            if self.cfg['use_ray_mask']:
                # prj_dict['mask']
                # (rfn,qn,rn,dn,-1)#reference views, query views,
                rfn, qn, rn, dn, _ = prj_dict["rgb"].shape
                mask = torch.ones([rfn, qn, rn, dn, 1], dtype=torch.float32, device=prj_dict["rgb"].device)
                outputs['ray_mask'] = torch.sum(mask.int(),0)>=self.cfg['ray_mask_view_num'] # qn,rn,dn,1
                outputs['ray_mask'] = torch.sum(outputs['ray_mask'],2)>self.cfg['ray_mask_point_num'] # qn,rn
                outputs['ray_mask'] = outputs['ray_mask'][...,0]
            
            if self.cfg['render_depth']:
                # qn,rn,dn
                outputs['render_depth'] = torch.sum(hit_prob_nr * new_que_depth, -1) # qn,rn

            if self.cfg['render_uncert']:
                # outputs['']
                # que_depth = qn, rn
                # pred_mean  = outputs["render_depth"]
                # dist_vals = que_depth
                # weights = hit_prob_nr
                # import ipdb;ipdb.set_trace()
                outputs['render_uncert'] = ((new_que_depth - outputs["render_depth"].unsqueeze(-1)).pow(2) * hit_prob_nr).sum(-1) + 1e-5
                
            if "perpoint_loss" in self.cfg and self.cfg["perpoint_loss"]:
                outputs["render_weights"] = hit_prob_nr
                outputs["render_dvals"] = que_depth
        return outputs

    def fine_render_impl(self, coarse_render_info, que_imgs_info, ref_imgs_info, is_train, is_perspec=False):
        # print("shape:", coarse_render_info['depth'].shape)
        fine_depth = sample_fine_depth(self.cfg, coarse_render_info['depth'], coarse_render_info['hit_prob'].detach(),
                                       que_imgs_info['depth_range'], self.cfg['fine_depth_sample_num'], is_train)
        if "ft_depth_range" in que_imgs_info:
            ft_depth_range = que_imgs_info["ft_depth_range"]
            # import ipdb;ipdb.set_trace()
            valid_depth = ft_depth_range[..., 0] >= self.cfg["min_depth"]
            invalid_depth = valid_depth.logical_not()        
            z_vals_2 = torch.empty_like(coarse_render_info['depth'])# 1, 512, 64
            N = coarse_render_info['depth'].shape[-1]
            z_vals_2[invalid_depth] = fine_depth[invalid_depth]
            # sample with in 3 sigma of the input depth, if it is valid
            if is_train:
                perturb = 1.0
            else:
                perturb = 0.0
            # import ipdb;ipdb.set_trace()

            z_vals_2[valid_depth] = sample_3sigma(ft_depth_range[valid_depth][:, 1], ft_depth_range[valid_depth][:, 2], N, perturb==0., self.cfg["min_depth"], self.cfg["max_depth"])

        else:
            z_vals_2 = fine_depth

        if torch.isnan(fine_depth).any():
            import ipdb;ipdb.set_trace()
        if torch.isnan(coarse_render_info['hit_prob']).any():
            import ipdb;ipdb.set_trace()
        # print("fine_depth.max():", fine_depth.max())
        # print("fine_depth.min():", fine_depth.min())
        # import ipdb;ipdb.set_trace()
        

        # qn, rn, fdn+dn
        if self.cfg['fine_depth_use_all']:
            que_depth = torch.sort(torch.cat([coarse_render_info['depth'], z_vals_2], -1), -1)[0]
        else:
            que_depth = torch.sort(z_vals_2, -1)[0]

        if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
            is_fine = False
        else:
            is_fine = True
            # pass
        
        #que_depth by coarse MLP
        outputs = self.render_by_depth(que_depth, que_imgs_info, ref_imgs_info, is_train, is_fine, is_perspec=is_perspec)

        # if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
        if "render_c2f_all" in self.cfg and self.cfg["render_c2f_all"]:
            colors_fine = outputs["colors_nr"]
            density_fine = outputs["density_nr"]
            # fine_depth: que_depth
            # coarse_depth: coarse_render_info["depth"]
            # import ipdb;ipdb.set_trace()
            z_vals = torch.cat([coarse_render_info["depth"], que_depth], dim=2)#1, 512, 64->1, 512, 128

            colors_all = torch.cat((coarse_render_info["colors"], colors_fine), dim=2)# 1, 512, 64, 3-> 1,512, 128, 3
            density_all = torch.cat((coarse_render_info["density"], density_fine), dim=2)# 1, 512, 64, -> 1,512, 128
            z_vals, indices = z_vals.sort()
            colors_all = torch.gather(colors_all, 2, indices.unsqueeze(-1).expand_as(colors_all))
            density_all = torch.gather(density_all, 2, indices)
            alpha_values = 1.0 - torch.exp(-torch.relu(density_all))
            hit_prob = alpha_values2hit_prob(alpha_values)
            pixel_colors = torch.sum(hit_prob.unsqueeze(-1)*colors_all,2)
            # return hit_prob, colors, pixel_colors, density
            # outputs["hit_prob"]
            # import ipdb;ipdb.set_trace()
            outputs['pixel_colors_nr'] = pixel_colors
            outputs['hit_prob_nr'] = hit_prob
            outputs["colors_nr"] = colors_all
            outputs["density_nr"] = density_all
            # import ipdb;ipdb.set_trace()
            if self.cfg['render_depth']:
                # qn,rn,dn
                outputs['render_depth'] = torch.sum(hit_prob * z_vals, -1) # qn,rn
            if self.cfg['render_uncert']:
                # outputs['']
                # que_depth = qn, rn
                # pred_mean  = outputs["render_depth"]
                # dist_vals = que_depth
                # weights = hit_prob_nr
                # import ipdb;ipdb.set_trace()
                outputs['render_uncert'] = ((z_vals - outputs["render_depth"].unsqueeze(-1)).pow(2) * hit_prob).sum(-1) + 1e-5
            # if "perpoint_loss" in self.cfg and self.cfg["perpoint_loss"]:
            #     outputs["render_weights"] = hit_prob_nr
            #     outputs["render_dvals"] = que_depth

        
        return outputs

    def merge_uniform_diner(self, diner_outputs, uniform_outputs):
        outputs = diner_outputs
        if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
            # colors_fine = outputs["colors_nr"]
            # density_fine = outputs["density_nr"]
            # fine_depth: que_depth
            # coarse_depth: coarse_render_info["depth"]
            # import ipdb;ipdb.set_trace()
            z_vals = torch.cat([diner_outputs["que_depth"], uniform_outputs['que_depth']], dim=2)#1, 512, 64->1, 512, 128
            colors_all = torch.cat((diner_outputs["colors_nr"], uniform_outputs['colors_nr']), dim=2)# 1, 512, 64, 3-> 1,512, 128, 3
            density_all = torch.cat((diner_outputs["density_nr"], uniform_outputs['density_nr']), dim=2)# 1, 512, 64, -> 1,512, 128
            z_vals, indices = z_vals.sort()
            colors_all = torch.gather(colors_all, 2, indices.unsqueeze(-1).expand_as(colors_all))
            density_all = torch.gather(density_all, 2, indices)
            alpha_values = 1.0 - torch.exp(-torch.relu(density_all))
            hit_prob = alpha_values2hit_prob(alpha_values)
            pixel_colors = torch.sum(hit_prob.unsqueeze(-1)*colors_all,2)
            # return hit_prob, colors, pixel_colors, density
            # outputs["hit_prob"]
            # import ipdb;ipdb.set_trace()
            outputs['pixel_colors_nr'] = pixel_colors
            outputs['hit_prob_nr'] = hit_prob
            outputs["colors_nr"] = colors_all
            outputs["density_nr"] = density_all
            # import ipdb;ipdb.set_trace()
            if self.cfg['render_depth']:
                # qn,rn,dn
                outputs['render_depth'] = torch.sum(hit_prob * z_vals, -1) # qn,rn
            if self.cfg['render_uncert']:
                # outputs['']
                # que_depth = qn, rn
                # pred_mean  = outputs["render_depth"]
                # dist_vals = que_depth
                # weights = hit_prob_nr
                # import ipdb;ipdb.set_trace()
                outputs['render_uncert'] = ((z_vals - outputs["render_depth"].unsqueeze(-1)).pow(2) * hit_prob).sum(-1) + 1e-5
            # if "perpoint_loss" in self.cfg and self.cfg["perpoint_loss"]:
            #     outputs["render_weights"] = hit_prob_nr
            #     outputs["render_dvals"] = que_depth
        return outputs
    #todo
    def render_impl(self, que_imgs_info, ref_imgs_info, is_train, is_perspec=False):
        #use_disp: false
        # import ipdb;ipdb.set_trace()
        if "diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"]:
            diner_que_depth, _ = sample_depth(self.cfg, que_imgs_info['coords'], self.cfg["n_candidates"], False, use_disp=False)

            diner_outputs = self.diner_render_by_depth(diner_que_depth, que_imgs_info, ref_imgs_info, is_train, False, is_perspec=is_perspec)

            if "N_uniform" in self.cfg and self.cfg["N_uniform"] > 0:
                uniform_que_depth, _ = sample_depth(self.cfg, que_imgs_info['coords'], self.cfg['depth_sample_num'], False, use_disp=True)            
                uniform_outputs = self.render_by_depth(uniform_que_depth, que_imgs_info, ref_imgs_info, is_train, False, is_perspec=is_perspec)
                uniform_outputs["que_depth"] = uniform_que_depth
                diner_outputs = self.merge_uniform_diner(diner_outputs, uniform_outputs)
            

            
            if "c2f" in self.cfg and self.cfg['c2f']:
                coarse_render_info = {'depth': diner_outputs["que_depth"], 'hit_prob': diner_outputs['hit_prob_nr']}
                # if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
                coarse_render_info["colors"] = diner_outputs["colors_nr"]
                coarse_render_info["density"] = diner_outputs["density_nr"]
            
                fine_outputs = self.fine_render_impl(coarse_render_info, que_imgs_info, ref_imgs_info, is_train, is_perspec=is_perspec)
                outputs = diner_outputs#{}
                for k, v in fine_outputs.items():
                    outputs[k + "_fine"] = v
                # import ipdb;ipdb.set_trace()
            else:
                outputs = {} #diner_outputs#{}
                for k, v in diner_outputs.items():
                    outputs[k + "_fine"] = v
        
        #diner_que_depth: 1, 512, 1000
        else:
            # [qn,rn,dn]
            # if self.cfg[""]
                # ft_precomputed_z_samples = que_imgs_info["ft_z_samples"]
                # que_depth, _ = sample_depth_with_guidance(self.cfg, que_imgs_info['coords'], self.cfg['depth_sample_num'], False, ft_depth_range, ft_precomputed_z_samples)
                # import ipdb;ipdb.set_trace()
            # else:
            que_depth, _ = sample_depth(self.cfg, que_imgs_info['coords'], self.cfg['depth_sample_num'], False, use_disp=self.cfg["use_disp"])
       
            # 2. #1, 2048, 64
            # import ipdb;ipdb.set_trace()
            if self.cfg["debug"]:
                # que_depth: qn, rn, dn
                # import ipdb;ipdb.set_trace()
                tmp_coords = que_imgs_info["coords"]#qn, rn, 2
                #  que_imgs_info["depth"]: qn, h, w, 1
                # print("que_imgs_info_depth.shape:",que_imgs_info["depth"].shape)
                que_depth = que_imgs_info["depth"][..., tmp_coords[..., 1].long(), tmp_coords[..., 0].long()] #qn, 1, h, w
                # qn, 1, qn, rn->qn, 1, rn
                que_depth = que_depth.squeeze(2).permute((0, 2, 1)) # qn, rn, 1
            # import ipdb;ipdb.set_trace()
            
            outputs = self.render_by_depth(que_depth, que_imgs_info, ref_imgs_info, is_train, False, is_perspec=is_perspec)

            if self.cfg['use_hierarchical_sampling']:
                coarse_render_info = {'depth': que_depth, 'hit_prob': outputs['hit_prob_nr']}
                # if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
                coarse_render_info["colors"] = outputs["colors_nr"]
                coarse_render_info["density"] = outputs["density_nr"]
                fine_outputs = self.fine_render_impl(coarse_render_info, que_imgs_info, ref_imgs_info, is_train, is_perspec=is_perspec)
                for k, v in fine_outputs.items():
                    outputs[k + "_fine"] = v
        
        return outputs
    
    def render(self, que_imgs_info, ref_imgs_info, is_train, is_perspec=False):
        if self.cfg["autoencoder"]:
            ref_img_feats, ae_outputs = self.image_encoder(ref_imgs_info['imgs'])
        else:
            ref_img_feats = self.image_encoder(ref_imgs_info['imgs'])
        
        ref_imgs_info['img_feats'] = ref_img_feats
        ref_imgs_info['ray_feats'] = self.vis_encoder(ref_imgs_info['ray_feats'], ref_img_feats)
        if is_train and self.cfg['use_self_hit_prob']:
            que_img_feats = self.image_encoder(que_imgs_info['imgs'])
            que_imgs_info['ray_feats'] = self.vis_encoder(que_imgs_info['ray_feats'], que_img_feats)

        ray_batch_num = self.cfg["ray_batch_num"]
        coords = que_imgs_info['coords']
        if 'ft_depth_range' in que_imgs_info and que_imgs_info['ft_depth_range'] is not None:
            ft_depth_range = que_imgs_info["ft_depth_range"]
            # ft_z_samples = que_imgs_info["ft_z_samples"]
            
        
        ray_num = coords.shape[1]
        render_info_all = {}
        for ray_id in range(0,ray_num,ray_batch_num):
            que_imgs_info['coords'] = coords[:,ray_id:ray_id+ray_batch_num]
            if 'ft_depth_range' in que_imgs_info and que_imgs_info['ft_depth_range'] is not None:
                # import ipdb;ipdb.set_trace()
                que_imgs_info['ft_depth_range'] = ft_depth_range[:,ray_id:ray_id+ray_batch_num]
                # que_imgs_info['ft_z_samples'] = ft_z_samples[:, ray_]
            # if depth_range is not None:
            #     import ipdb;ipdb.set_trace()

            # print("ray_id:", ray_id)
            # if ray_id / 2048 == 61: #256*512/2048= 256/4=64
            #     self.cfg["start_debug"] = True
            # else:
            #     self.cfg["start_debug"] = False         
            # rgb_test = np.uint8(que_imgs_info["imgs"].permute((0, 2, 3, 1)).data.cpu().numpy()*255)[0]
            # # rgb_test = rgb_test
            # import cv2
            # cv2.imwrite("rgb_test2.jpg", rgb_test)
            render_info = self.render_impl(que_imgs_info,ref_imgs_info,is_train, is_perspec)
            output_keys = [k for k in render_info.keys() if is_train or (not k.startswith('hit_prob'))]
            for k in output_keys:
                v = render_info[k]
                if k not in render_info_all:
                    render_info_all[k]=[]
                render_info_all[k].append(v)

        for k, v in render_info_all.items():
            render_info_all[k]=torch.cat(v,1)
        if self.cfg["autoencoder"]:
            render_info_all["ae_outputs"] = ae_outputs# = self.image_encoder(ref_imgs_info['imgs'])
        return render_info_all

class NeuralRayGenRenderer(NeuralRayBaseRenderer):
    default_cfg={
        'init_net_type': 'depth',
        'init_net_cfg': {},

        'use_depth_loss': False,
        'depth_loss_coords_num': 8192,
    }
    def __init__(self, cfg):
        cfg={**self.default_cfg,**cfg}
        super().__init__(cfg)
        #todo?
        self.init_net=name2init_net[self.cfg['init_net_type']](self.cfg)

    def render_call(self, que_imgs_info, ref_imgs_info, is_train, src_imgs_info=None, is_perspec=False):
        # if self.cfg["autoencoder"]:
        #     ref_imgs_info['ray_feats'], ray_ae_outputs = self.init_net(ref_imgs_info, src_imgs_info, is_train)
        # else:
        #     ref_imgs_info['ray_feats'] = self.init_net(ref_imgs_info, src_imgs_info, is_train)
        ret = self.init_net(ref_imgs_info, src_imgs_info, is_train)
        ref_imgs_info['ray_feats'] = ret["ray_feats"]
        ref_imgs_info['mvs_depth'] = ret['mvs_depth']
        if "uncert_tune" in self.cfg and self.cfg["uncert_tune"]:
            ref_imgs_info['mvs_uncert'] = ret['mvs_uncert']
        # import ipdb;ipdb.set_trace()
        if "backface_culling" in self.cfg and self.cfg["backface_culling"]:
            ref_imgs_info['mvs_normal'] = depth2normal(ref_imgs_info, self.spt_utils)
        
        # import ipdb;ipdb.set_trace()

        outputs = self.render(que_imgs_info, ref_imgs_info, is_train, is_perspec)
        # if self.cfg["autoencoder"]:    
        #     outputs["ray_ae_outputs"] = ray_ae_outputs

        return outputs
    def gen_depth_loss_coords(self,h,w,device):
        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).reshape(-1, 2).to(device)
        num = self.cfg['depth_loss_coords_num']
        idxs = torch.randperm(coords.shape[0])
        idxs = idxs[:num]
        coords = coords[idxs]
        return coords

    def predict_mean_for_depth_loss(self, ref_imgs_info):
        ray_feats = ref_imgs_info['ray_feats'] # rfn, f,h',w'
        ref_imgs = ref_imgs_info['imgs'] # rfn,3,h,w
        rfn, _, h, w = ref_imgs.shape
        coords = self.gen_depth_loss_coords(h, w, ref_imgs.device) # pn,2
        coords = coords.unsqueeze(0).repeat(rfn,1,1) # rfn,pn,2
        batch_num = self.cfg['depth_loss_coords_num']
        pn = coords.shape[1]
        coords_dist_mean, coords_dist_mean_2, coords_dist_mean_fine, coords_dist_mean_fine_2 = [], [], [], []
        for ci in range(0, pn, batch_num):
            coords_ = coords[:,ci:ci+batch_num]
        
            # mask_ = torch.ones(coords_.shape[:2], dtype=torch.float32, device=ref_imgs.device)
            coords_ray_feats_ = interpolate_feature_map(ray_feats, coords_, h, w) # rfn,pn,f
            coords_dist_mean_ = self.dist_decoder.predict_mean(coords_ray_feats_)  # rfn,pn
            coords_dist_mean_2.append(coords_dist_mean_[..., 1])
            coords_dist_mean_ = coords_dist_mean_[..., 0]

            coords_dist_mean.append(coords_dist_mean_)
            if self.cfg['use_hierarchical_sampling']:
                if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
                    pass
                else:
                    coords_dist_mean_fine_ = self.fine_dist_decoder.predict_mean(coords_ray_feats_)
                    coords_dist_mean_fine_2.append(coords_dist_mean_fine_[..., 1])
                    coords_dist_mean_fine_ = coords_dist_mean_fine_[..., 0]  # use 0 for depth supervision
                    coords_dist_mean_fine.append(coords_dist_mean_fine_)

        coords_dist_mean = torch.cat(coords_dist_mean, 1)
        outputs = {'depth_mean': coords_dist_mean, 'depth_coords': coords}
        if len(coords_dist_mean_2)>0:
            coords_dist_mean_2 = torch.cat(coords_dist_mean_2, 1)
            outputs['depth_mean_2'] = coords_dist_mean_2
        
        if self.cfg['use_hierarchical_sampling']:
            if "one_mlp" in self.cfg and self.cfg["one_mlp"]:
                pass
            else:
                
                coords_dist_mean_fine = torch.cat(coords_dist_mean_fine, 1)
                outputs['depth_mean_fine'] = coords_dist_mean_fine
                if len(coords_dist_mean_fine_2)>0:
                    coords_dist_mean_fine_2 = torch.cat(coords_dist_mean_fine_2, 1)
                    outputs['depth_mean_fine_2'] = coords_dist_mean_fine_2
        return outputs

    def forward(self,data, is_perspec=False):
        ref_imgs_info = data['ref_imgs_info'].copy()
        que_imgs_info = data['que_imgs_info'].copy()
        is_train = 'eval' not in data
        src_imgs_info = data['src_imgs_info'].copy() if 'src_imgs_info' in data else None
        render_outputs = self.render_call(que_imgs_info, ref_imgs_info, is_train, src_imgs_info, is_perspec=is_perspec)
        
        if (self.cfg['use_depth_loss'] and 'true_depth' in ref_imgs_info) or (not is_train):
            render_outputs.update(self.predict_mean_for_depth_loss(ref_imgs_info))
        return render_outputs

class NeuralRayFtRenderer(NeuralRayBaseRenderer):
    default_cfg={
        # scene
        # 'database_name': 'nerf_synthetic/lego/black_400',
        # "database_split": 'val_all',

        # input config
        # "ref_pad_interval": 16,
        # "use_consistent_depth_range": True,

        # training related
        'gen_cfg': None, # 'configs/train/gen/ft_lr_neuray_lego.yaml'
        "use_validation": True,
        "validate_initialization": True, # visualize rendered images of inited neuray on the val set
        # "init_view_num": 2, # number of neighboring views used in initialization: this should be consistent with the number used in generalization model
        "init_src_view_num": 1,

        # neighbor view selection in training
        'train_include_self': False,

        "include_self_prob": 0.01,
        # "neighbor_view_num": 2,  # number of neighboring views
        # "neighbor_pool_ratio": 2,
        "train_ray_num": 512,
        "foreground_ratio": 1.0,

        # used in train from scratch
        'ray_feats_res': [64, 128], # size of raw visibility feature G': H=200,W=200
        'ray_feats_dim': 32, # channel number of raw visibility feature G'
        'depth_guided_ray_sampling': False,
        "has_depth": True,

    }
    def __init__(self, cfg):
        cfg = {**self.default_cfg,**cfg}
        cfg["ray_feats_res"] = [cfg["height"]//8, cfg["width"]//8]
        print("cfg.ray_feats_res:", cfg["ray_feats_res"])
        super().__init__(cfg)
        self.cached = False
        if cfg["dataset_name"] == "m3d":
            if cfg["database_name"] =="replica_wide":
                from data_readers.replica_wide import ReplicaWideDataset
                dataset = ReplicaWideDataset(
                    cfg=cfg)
            elif cfg["database_name"] == "m3d":
                mode="test"
                if cfg["use_lmdb"]:
                    dataset = HabitatImageGeneratorFT_LMDB(
                        args=cfg,
                        split=mode,
                        seq_len=cfg["seq_len"],
                        reference_idx=cfg["reference_idx"],
                        full_width=cfg["width"],
                        full_height=cfg["height"],
                        m3d_dist=cfg["m3d_dist"]
                    )
                else:
                    dataset = HabitatImageGeneratorFT(
                        args=cfg,
                        split=mode,
                        seq_len=cfg["seq_len"],
                        reference_idx=cfg["reference_idx"],
                        full_width=cfg["width"],
                        full_height=cfg["height"],
                        m3d_dist=cfg["m3d_dist"]
                    )
            # import ipdb;ipdb.set_trace()
            data = dataset.__getitem__(cfg["data_idx"])
            import cv2
            img = data["rgb_panos"]
            rgb = img[1]
            rgb = np.uint8(rgb*255)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)        
            cv2.imwrite("debug_1.jpg", rgb)      
            
            rgb = img[0]
            rgb = np.uint8(rgb*255)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)        
            cv2.imwrite("debug_0.jpg", rgb)
            rgb = img[2]
            rgb = np.uint8(rgb*255)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)        
            cv2.imwrite("debug_2.jpg", rgb)
        
            database = M3DDatabase(cfg, data)
        else:
            # dataset = 
            database = ReplicaDatabase(cfg)

        self.database = database#parse_database_name(self.cfg['database_name'])
        self.ref_ids = [0, 2]        
        self.val_ids = np.asarray([1])
        self.ref_ids = np.asarray(self.ref_ids)

        # build imgs_info
        ref_imgs_info = build_imgs_info(self.database, self.ref_ids, replace_none_depth=False, has_depth=self.cfg["has_depth"]) #???

        self.ref_imgs_info = imgs_info_to_torch(ref_imgs_info)
        if self.cfg['use_validation']:
            val_imgs_info = build_imgs_info(self.database, self.val_ids, -1, True, has_depth=self.cfg["has_depth"])
            self.val_imgs_info = imgs_info_to_torch(val_imgs_info)
            self.val_num = len(self.val_ids)
        # init from generalization model


        self._initialization()
        if self.cfg["depth_guided_ray_sampling"]:
            self.precomputed_z_samples = precompute_quadratic_samples(self.cfg["min_depth"], self.cfg["max_depth"], self.cfg["quadratic_n_samples"])


        # after initialization, we check the correctness of rendered images
        if self.cfg['use_validation'] and self.cfg['validate_initialization']:
            print('init validation rendering ...')
            Path(f'data/vis_val/{self.cfg["name"]}').mkdir(exist_ok=True, parents=True)
            self.eval()
            self.cuda()
            for vi in tqdm(range(self.val_num)):
                outputs = self.validate_step(vi, step=0)
                key_name = 'pixel_colors_nr_fine' if self.cfg['use_hierarchical_sampling'] or ("diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"])else 'pixel_colors_nr'
                img_gt = self.val_imgs_info['imgs'][vi] # 3,h,w
                _, h, w = img_gt.shape
                img_gt = color_map_backward(img_gt.permute(1,2,0).numpy())
                rgb_pr = outputs[key_name].reshape(h, w, 3).cpu().numpy()
                img_pr = color_map_backward(rgb_pr)
                imsave(f'data/vis_val/{self.cfg["name"]}/init-{vi}.jpg',concat_images_list(img_gt,img_pr))
                # outputs, data



    def _init_by_cost_volume(self, ref_id, init_net):
        ref_imgs_info = imgs_info_slice(self.ref_imgs_info,torch.from_numpy(np.asarray([self.ref_ids.tolist().index(ref_id)])).long())        
        src_num = self.cfg['init_src_view_num']
        src_index = list(set((range(len(self.ref_ids))))-set([self.ref_ids.tolist().index(ref_id)]))
        src_imgs_info = imgs_info_slice(self.ref_imgs_info, torch.from_numpy(np.asarray(src_index)).long())
        with torch.no_grad():
            ret = init_net(to_cuda(ref_imgs_info), to_cuda(src_imgs_info), False)
            ray_feats_cur = ret["ray_feats"]
            mvs_depth_cur = ret['mvs_depth']
            if self.cfg["render_uncert"]:
                mvs_var_cur = ret['mvs_uncert']

        ray_feats_cur = ray_feats_cur.detach().cpu()
        mvs_depth_cur = mvs_depth_cur.detach().cpu()

        if self.cfg["render_uncert"]:
            mvs_var_cur = mvs_var_cur.detach().cpu()
            return ray_feats_cur, mvs_depth_cur, mvs_var_cur
        else:
            return ray_feats_cur, mvs_depth_cur

    def _init_raw_visibility_features(self, ref_id, init_net):
        if isinstance(init_net, CostVolumeInitNet):

            if self.cfg["render_uncert"]:
                ray_feats_cur, mvs_depth_cur, mvs_var_cur = self._init_by_cost_volume(ref_id, init_net)
            else:
                ray_feats_cur, mvs_depth_cur = self._init_by_cost_volume(ref_id, init_net)

        else:
            raise NotImplementedError
        if self.cfg["render_uncert"]:
            return ray_feats_cur, mvs_depth_cur, mvs_var_cur

        else:
            return ray_feats_cur, mvs_depth_cur

    def _initialization(self):
        self.ray_feats = nn.ParameterList()
        if self.cfg['gen_cfg'] is not None:
            # load generalization model
            gen_cfg = load_cfg(self.cfg['gen_cfg'])
            name = gen_cfg['name']
            # 1. keep gen the same resolution with ft
            gen_cfg['width'] = self.cfg["width"]
            gen_cfg["height"] = self.cfg["height"]
            gen_cfg["dataset_name"] = self.cfg["dataset_name"]    
            # import ipdb;ipdb.set_trace()
            gen_cfg["handle_distort"] = self.cfg['handle_distort']
            gen_cfg["handle_distort_all"] = self.cfg['handle_distort_all']
            gen_cfg["handle_distort_input_all"] = self.cfg['handle_distort_input_all']
            gen_cfg["mono_uncert_tune"] = self.cfg["mono_uncert_tune"]
            gen_cfg["with_sin"] = self.cfg["with_sin"]
            gen_cfg["wo_mono_feat"] = self.cfg["wo_mono_feat"]
            print("gen-cfg.dataset_name:", gen_cfg["dataset_name"])
            # 2. remember to revise the resolution of patchsize for tangent projection
            ckpt = torch.load(f'data/model/{name}/model.pth')
            # import ipdb;ipdb.set_trace()

            gen_renderer = NeuralRayGenRenderer(gen_cfg).cuda()

            # import ipdb;ipdb.set_trace()
            gen_renderer.load_state_dict(ckpt['network_state_dict'])
            # import ipdb;ipdb.set_trace()
            gen_renderer = gen_renderer.eval()
            # # #???
            # from network.omni_mvsnet.pipeline3_model import load_checkpoint
            # gen_renderer.init_net.mvsnet.d_net = load_checkpoint(gen_cfg["DNET_ckpt"], gen_renderer.init_net.mvsnet.d_net, "model_state_dict")#"dnet")
            # gen_renderer.init_net.mvsnet.d_net.eval()

            # gen_renderer.init_net.mvsnet.load_state_dict(torch.load(gen_cfg["mvsnet_pretrained_path"], map_location=gen_cfg["device"])['model_state_dict'])
            # gen_renderer.init_net.mvsnet.eval()


            # init from generalization model
            print('initialization ...')
            self.mvs_depths = []
            self.mvs_uncert = []

            for ref_id in tqdm(self.ref_ids):
                print("ref_id:", ref_id)
                # self.ray_feats is ordered.
                # ray_feats, mvs_depths = self._init_raw_visibility_features(ref_id, gen_renderer.init_net)
                if self.cfg["render_uncert"]:
                    ray_feats, mvs_depths, mvs_uncert = self._init_raw_visibility_features(ref_id, gen_renderer.init_net)
    
                else:
                    ray_feats, mvs_depths = self._init_raw_visibility_features(ref_id, gen_renderer.init_net)

                # self.ray_feats.append(nn.Parameter(self._init_raw_visibility_features(ref_id, gen_renderer.init_net)))
                self.ray_feats.append(nn.Parameter(ray_feats)) #self._init_raw_visibility_features(ref_id, gen_renderer.init_net)))
                self.mvs_depths.append(mvs_depths)
                if self.cfg["render_uncert"]:
                    self.mvs_uncert.append(mvs_uncert)

            # init other parameters
            self.vis_encoder.load_state_dict(gen_renderer.vis_encoder.state_dict())
            self.dist_decoder.load_state_dict(gen_renderer.dist_decoder.state_dict())
            self.agg_net.load_state_dict(gen_renderer.agg_net.state_dict())
            # if self.cfg["fix_coarse"]:
            #     for param in self.agg_net.parameters():
            #         param.requires_grad = False
            #     self.agg_net.eval()
            self.sph_fitter.load_state_dict(gen_renderer.sph_fitter.state_dict())
            # import ipdb;ipdb.set_trace()
            self.image_encoder.load_state_dict(gen_renderer.image_encoder.state_dict())            
            # import ipdb;ipdb.set_trace()
            if self.cfg['use_hierarchical_sampling']:      
                if "one_mlp" in self.cfg and self.cfg["one_mlp"]:          
                    pass
                else:
                    self.fine_dist_decoder.load_state_dict(gen_renderer.fine_dist_decoder.state_dict())
                    self.fine_agg_net.load_state_dict(gen_renderer.fine_agg_net.state_dict())
        else:
            print('init from scratch !')
            fh, fw = self.cfg['ray_feats_res']
            dim = self.cfg['ray_feats_dim']
            ref_num = len(self.ref_ids)
            for k in range(ref_num):
                self.ray_feats.append(nn.Parameter(torch.randn(1,dim,fh,fw)))

    def slice_imgs_info(self, ref_idx, val_idx, is_train):
        # prepare ref imgs info
        # print("slice_imgs_info, ref_idx:" ,ref_idx)
        # if "my_debug" in self.cfg and self.cfg["my_debug"]:
        #     ref_imgs_info = self.ref_imgs_info.copy()
        # else:
        #re-ordered
        ref_imgs_info = imgs_info_slice(self.ref_imgs_info, torch.from_numpy(np.asarray(ref_idx)).long()) #

        ref_imgs_info = to_cuda(ref_imgs_info)
        ref_imgs_info['ray_feats'] = torch.cat([self.ray_feats[ref_i] for ref_i in ref_idx], 0)
        if self.cfg['gen_cfg'] is not None:
            ref_imgs_info['mvs_depth'] = torch.cat([self.mvs_depths[ref_i] for ref_i in ref_idx], 0).cuda()
            if self.cfg['render_uncert']:
                ref_imgs_info['mvs_uncert'] = torch.cat([self.mvs_uncert[ref_i] for ref_i in ref_idx], 0).cuda()
        # prepare que_imgs_info
        
        if is_train:
            # import ipdb;ipdb.set_trace()
            if "my_debug" in self.cfg and self.cfg["my_debug"]:
                que_imgs_info = imgs_info_slice(self.ref_imgs_info, torch.from_numpy(np.asarray([val_idx])).long())
                qn, _, hn, wn = que_imgs_info['imgs'].shape
                coords = np.stack(np.meshgrid(np.arange(wn), np.arange(hn)), -1)
                coords = coords.reshape([1, -1, 2]).astype(np.float32)
            
            else:
                #bug?, ref_imgs_info是经过ref_idx重新排序的, val_idx还是按照[0, 1]原有的次序(self.ref_imgs_info)取出的
                que_imgs_info = imgs_info_slice(self.ref_imgs_info, torch.from_numpy(np.asarray([val_idx])).long())
                qn, _, hn, wn = que_imgs_info['imgs'].shape
                que_mask_cur = que_imgs_info['masks'][0, 0].cpu().numpy() > 0
                # print()
                # import ipdb;ipdb.set_trace()
                # print("que_mask_cur:", que_mask_cur)
                coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], self.cfg['foreground_ratio']).reshape([1, -1, 2])

        else:            
            que_imgs_info = imgs_info_slice(self.val_imgs_info, torch.from_numpy(np.asarray([val_idx])).long())
            qn, _, hn, wn = que_imgs_info['imgs'].shape
            coords = np.stack(np.meshgrid(np.arange(wn), np.arange(hn)), -1) #W,H
            coords = coords.reshape([1, -1, 2]).astype(np.float32)

        que_imgs_info['coords'] = torch.from_numpy(coords)
        if self.cfg["depth_guided_ray_sampling"] and is_train:
            # import ipdb;ipdb.set_trace();
            target_d = self.mvs_depths[val_idx] #1, 1, h, w
            uncert_d = self.mvs_uncert[val_idx] # 1, 1, h, w
            # target_d = target_d[coords]
            target_d = target_d[:, :, coords[..., 1], coords[..., 0]].squeeze()
            uncert_d = uncert_d[:, :, coords[..., 1], coords[..., 0]].squeeze()
            std = torch.sqrt(uncert_d)
            if "ft_fixed_sigma" in self.cfg:
                std = self.cfg["ft_fixed_sigma"]               
            # else:
            #     pass

            #todo, std: fixed, or add a parameter
            depth_range = precompute_depth_sampling(target_d, std)
            que_imgs_info['ft_depth_range'] = depth_range.unsqueeze(0)
            # que_imgs_info['ft_z_samples'] = self.precomputed_z_samples

        que_imgs_info = to_cuda(que_imgs_info)
        if is_train and self.cfg['use_self_hit_prob']:
            #self.ray_feats: original order
            que_imgs_info['ray_feats'] = self.ray_feats[val_idx]

        if is_train:
            if self.cfg['gen_cfg'] is not None:
                que_imgs_info['mvs_depth'] = self.mvs_depths[val_idx]
                if self.cfg["render_uncert"]:
                    que_imgs_info['mvs_uncert'] = self.mvs_uncert[val_idx]
        return ref_imgs_info, que_imgs_info

    def validate_step(self, val_idx, step):# val_idx: 0
        ref_idx = range(len(self.ref_ids))
        # ref_idx = self.val_dist_idx[val_idx][:self.cfg['neighbor_view_num']]
        ref_imgs_info, que_imgs_info = self.slice_imgs_info(ref_idx, val_idx, False)        
        with torch.no_grad():
            render_outputs = self.render(que_imgs_info.copy(), ref_imgs_info.copy(), False)

        ref_imgs_info.pop('ray_feats')
        h, w = ref_imgs_info["imgs"].shape[2:4]
        render_outputs.update({'ref_imgs_info': ref_imgs_info, 'que_imgs_info': que_imgs_info})
        
        return render_outputs

    def train_step(self):
        # select neighboring views for training
        que_i = np.random.randint(0,len(self.ref_ids)) # example: 0 ([0, 1])

        if self.cfg["ft_include_self"]:
            if np.random.random() > self.cfg['include_self_prob']:
                ref_idx = list(set(range(len(self.ref_ids)))-set([que_i])) #[0, 1]
            else:              
                ref_idx = list(range(len(self.ref_ids)))
                np.random.shuffle(ref_idx)
                # import ipdb;ipdb.set_trace()
        else:
            ref_idx = list(set(range(len(self.ref_ids)))-set([que_i])) #[0, 1]

        ref_imgs_info, que_imgs_info = self.slice_imgs_info(ref_idx, que_i, True)
        # if self.cfg["depth_guided_ray_sampling"]:
        #     depth_range = self.depth_range
        # else:
        #     depth_range = None

        # if self.cfg["use_precomputed_z_samples"]:
        #     precomputed_z_samples = self.precomputed_z_samples
        # else:
        #     precomputed_z_samples = None
        # import ipdb;ipdb.set_trace()
        # que_imgs_info[""]
        render_outputs = self.render(que_imgs_info.copy(), ref_imgs_info.copy(), True) #?
        # import ipdb;ipdb.set_trace()

        # clear some values for outputs
        # ref_imgs_info.pop('ray_feats')
        # que_imgs_info.pop('ray_feats')
        if 'img_feats' in ref_imgs_info: ref_imgs_info.pop('img_feats')
        if 'img_feats' in que_imgs_info: que_imgs_info.pop('img_feats')
        render_outputs.update({'que_imgs_info': que_imgs_info, 'ref_imgs_info': ref_imgs_info})
        render_outputs["que_id"] = que_i
        #for debug
        render_outputs["all_info"] = self.ref_imgs_info
        return render_outputs

    

    def render_pose(self, render_imgs_info):
        print("render_pose")
        # this function is used in rendering from arbitrary poses
        render_pose = render_imgs_info['w2c'].cpu().numpy()
        ref_poses = self.ref_imgs_info['w2c'].cpu().numpy()
        ref_idx = np.asarray(range(len(self.ref_ids)))
        ref_imgs_info = to_cuda(imgs_info_slice(self.ref_imgs_info, torch.from_numpy(ref_idx).long()))
        ref_imgs_info['ray_feats'] = torch.cat([self.ray_feats[ref_i] for ref_i in ref_idx], 0)
        with torch.no_grad():
            render_outputs = self.render(render_imgs_info, ref_imgs_info, False) 
        return render_outputs

    def render_cube_pose(self, render_imgs_info):
        print("render_pose")
        # this function is used in rendering from arbitrary poses
        render_pose = render_imgs_info['poses'].cpu().numpy()
        ref_poses = self.ref_imgs_info['w2c'].cpu().numpy()
        ref_idx = np.asarray(range(len(self.ref_ids)))
        ref_imgs_info = to_cuda(imgs_info_slice(self.ref_imgs_info, torch.from_numpy(ref_idx).long()))
        ref_imgs_info['ray_feats'] = torch.cat([self.ray_feats[ref_i] for ref_i in ref_idx], 0)
        with torch.no_grad():
            render_outputs = self.render(render_imgs_info, ref_imgs_info, False, is_perspec=True)            
        return render_outputs
    def forward(self, data, is_perspec=False):
        index = data['index']  
        # if "my_debug" in self.cfg and self.cfg["my_debug"]:
        #     return self.validate_step(0)

        is_train = 'eval' not in data
        if 'eval' in data:
            step = data["step"]

        if is_train:
            return self.train_step()
        else:
            return self.validate_step(index, step)

name2network={
    'neuray_gen': NeuralRayGenRenderer,
    'neuray_ft': NeuralRayFtRenderer,
}