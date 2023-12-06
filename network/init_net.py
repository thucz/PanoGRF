import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inplace_abn import ABN
import math
# from network.mvsnet.modules import depth_regression
# from network.mvsnet.mvsnet import MVSNet, load_ckpt
from torchvision import transforms
import cv2

from network.omni_mvsnet.pipeline3_model import FullPipeline
from network.ops import interpolate_feats, masked_mean_var, ResUNetLight, conv3x3, ResidualBlock, conv1x1
from network.resnet_erp_tp import ResUNetLight_ERP_TP
from network.render_ops import project_points_ref_views
import sys
sys.path.append("./UniFuse-Unidirectional-Fusion/UniFuse")
# from datasets.util import Equirec2Cube
# load checkpoint
def load_checkpoint(fpath, model, model_name):
    # import ipdb;ipdb.set_trace()
    ckpt = torch.load(fpath, map_location='cpu')
        # import ipdb;ipdb.set_trace()
    if model_name in ckpt:
        ckpt = ckpt[model_name]
    
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v
   
    model.load_state_dict(load_dict)
    return model

def e2c_process(panos_small, e2c_instance):      
    #panos_small = panos_small.reshape(batch_size, seq_len, height, width, 3)
    batch_size = panos_small.shape[0]
    pano_cube_all = []
    for i in range(batch_size):
        # pano_cube_seq = []
        cube = e2c_instance.run(panos_small[i].data.cpu().numpy())
        # pano_cube_seq.append(cube)
        pano_cube_all.append(cube)
    pano_cube_all = np.array(pano_cube_all)
    pano_cube_all = torch.from_numpy(pano_cube_all)#.to(self.args["device"])
    return pano_cube_all
def normalize_input(panos_small, panos_small_cube, normalize_fn):      
    batch_size = panos_small.shape[0]
    normalized_equi = []
    normalized_cube = []
    for i in range(batch_size):
        normalized_equi.append(normalize_fn(panos_small[i]).data.cpu().numpy())
        normalized_cube.append(normalize_fn(panos_small_cube[i]).data.cpu().numpy())

    # normalized_equi = np.array(normalized_equi)
    normalized_equi = torch.from_numpy(np.array(normalized_equi))#.to(self.args["device"])
    normalized_cube = torch.from_numpy(np.array(normalized_cube))#.to(self.args["device"])
    return normalized_equi, normalized_cube

# def depth2pts3d(depth, ref_Ks, ref_poses):
#     rfn, dn, h, w = depth.shape
#     coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).float().to(depth.device)
#     coords = coords[:, :, (1, 0)]
#     coords = coords.unsqueeze(0)  # 1,h,w,2
#     coords = torch.cat([coords, torch.ones([1, h, w, 1], dtype=torch.float32, device=depth.device)], -1).unsqueeze(
#         -2)  # 1,h,w,1,3
#     # rfn,h,w,dn,1 1,h,w,1,3
#     pts3d = depth.permute(0, 2, 3, 1).unsqueeze(-1) * coords  # rfn,h,w,dn,3
#     pts3d = pts3d.reshape(rfn, h * w * dn, 3).permute(0, 2, 1)  # rfn,3,h*w*dn
#     pts3d = torch.inverse(ref_Ks) @ pts3d  # rfn
#     R = ref_poses[:, :, :3].permute(0, 2, 1)  # rfn,3,3
#     t = -R @ ref_poses[:, :, 3:]  # rfn,3,1
#     pts3d = R @ pts3d + t  # rfn,3,h*w*dn
#     return pts3d.permute(0, 2, 1)  # rfn,h*w*dn,3

# def get_diff_feats(ref_imgs_info, depth_in):
#     imgs = ref_imgs_info['imgs']  # rfn,3,h,w
#     depth_range = ref_imgs_info['depth_range']
#     near = depth_range[:, 0][:, None, None]  # rfn,1,1
#     far = depth_range[:, 1][:, None, None]  # rfn,1,1
#     near_inv, far_inv = -1 / near[..., None], -1 / far[..., None]
#     depth_in = depth_in * (far_inv - near_inv) + near_inv
#     depth = -1 / depth_in
#     rfn, _, h, w = imgs.shape

#     pts3d = depth2pts3d(depth, ref_imgs_info['Ks'], ref_imgs_info['poses'])
#     _, pts2d, pts_dpt_prj, valid_mask = project_points_ref_views(ref_imgs_info, pts3d.reshape(-1, 3))   # [rfn,rfn*h*w,2] [rfn,rfn*h*w] [rfn,rfn*h*w,1]
#     pts_dpt_int = interpolate_feats(depth, pts2d, padding_mode='border', align_corners=True)         # rfn,rfn*h*w,1
#     pts_rgb_int = interpolate_feats(imgs, pts2d, padding_mode='border', align_corners=True)          # rfn,rfn*h*w,3

#     rgb_diff = torch.abs(pts_rgb_int - imgs.permute(0, 2, 3, 1).reshape(1, rfn * h * w, 3))  # rfn,rfn*h*w,3

#     pts_dpt_int = torch.clamp(pts_dpt_int, min=1e-5)
#     pts_dpt_prj = torch.clamp(pts_dpt_prj, min=1e-5)
#     dpt_diff = torch.abs(-1 / pts_dpt_int + 1 / pts_dpt_prj)  # rfn,rfn*h*w,1
#     near_inv, far_inv = -1 / near, -1 / far
#     dpt_diff = dpt_diff / (far_inv - near_inv)
#     dpt_diff = torch.clamp(dpt_diff, max=1.5)

#     valid_mask = valid_mask.float().unsqueeze(-1)
#     dpt_mean, dpt_var = masked_mean_var(dpt_diff, valid_mask, 0)  # 1,rfn,h,w,1
#     rgb_mean, rgb_var = masked_mean_var(rgb_diff, valid_mask, 0)  # 1,rfn*h*w,3
#     dpt_mean = dpt_mean.reshape(rfn, h, w, 1).permute(0, 3, 1, 2)  # rfn,1,h,w
#     dpt_var = dpt_var.reshape(rfn, h, w, 1).permute(0, 3, 1, 2)  # rfn,1,h,w
#     rgb_mean = rgb_mean.reshape(rfn, h, w, 3).permute(0, 3, 1, 2)  # rfn,3,h,w
#     rgb_var = rgb_var.reshape(rfn, h, w, 3).permute(0, 3, 1, 2)  # rfn,3,h,w
#     return torch.cat([rgb_mean, rgb_var, dpt_mean, dpt_var], 1)

def extract_depth_for_init_impl(args,depth):
    rfn, _, h, w = depth.shape

    near = args["min_depth"]#depth_range[:, 0][:, None, None, None]  # rfn,1,1,1
    far = args["max_depth"]#depth_range[:, 1][:, None, None, None]  # rfn,1,1,1
    near_inv = -1 / near
    far_inv = -1 / far
    depth = torch.clamp(depth, min=1e-5)
    depth = -1 / depth
    depth = (depth - near_inv) / (far_inv - near_inv)
    depth = torch.clamp(depth, min=0, max=1.0) #归一化
    #disparity
    return depth

def extract_depth_for_init(ref_imgs_info):
    # depth_range = ref_imgs_info['depth_range']  # rfn,2
    depth = ref_imgs_info['depth']  # rfn,1,h,w
    return extract_depth_for_init_impl(depth_range, depth)

# class DepthInitNet(nn.Module):
#     default_cfg={}
#     def __init__(self,cfg):
#         super().__init__()
#         self.cfg={**self.default_cfg,**cfg}
#         self.res_net = ResEncoder()
#         self.depth_skip = nn.Sequential(
#             nn.Conv2d(1, 8, 2, 2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 16, 2, 2)
#         )
#         self.conv_out=nn.Conv2d(16+32,32,1,1)

#     def forward(self, ref_imgs_info, src_imgs_info, is_train):
#         depth = extract_depth_for_init(ref_imgs_info)
#         imgs = ref_imgs_info['imgs']
#         diff_feats = get_diff_feats(ref_imgs_info,depth)
#         # imgs [b,3,h,w] depth [b,1,h,w] diff_feats [b,8,h,w]
#         feats = self.res_net(torch.cat([imgs, depth, diff_feats], 1))
#         depth_feats = self.depth_skip(depth)
#         return self.conv_out(torch.cat([depth_feats, feats],1))

# def construct_project_matrix(x_ratio, y_ratio, Ks, poses):
#     rfn = Ks.shape[0]
#     scale_m = torch.tensor([x_ratio, y_ratio, 1.0], dtype=torch.float32, device=Ks.device)
#     scale_m = torch.diag(scale_m)
#     ref_prj = scale_m[None, :, :] @ Ks @ poses  # rfn,3,4
#     pad_vals = torch.zeros([rfn, 1, 4], dtype=torch.float32, device=ref_prj.device)
#     pad_vals[:, :, 3] = 1.0
#     ref_prj = torch.cat([ref_prj, pad_vals], 1)  # rfn,4,4
#     return ref_prj


def construct_input_data(cfg, ref_imgs_info, src_imgs_info):
    # import ipdb;ipdb.set_trace()
    pano_ref = F.interpolate(ref_imgs_info["imgs"], (cfg["depth_height"], cfg["depth_width"]), mode='bilinear')#2, _, H, W
    pano_src = F.interpolate(src_imgs_info["imgs"], (cfg["depth_height"], cfg["depth_width"]), mode='bilinear') #2, _, H, W
    
    # import ipdb;ipdb.set_trace()
    panos = torch.stack([torch.stack([pano_src[idx], pano_ref[idx]], dim=0) \
        for idx in range(len(pano_ref))], dim=0)    
    rot_ref = ref_imgs_info["rots"]#2, 
    rot_src = src_imgs_info["rots"]#2, 
    rots = torch.stack([torch.stack([rot_src[idx], rot_ref[idx]], dim=0) \
        for idx in range(len(pano_ref))], dim=0)
    
    trans_ref = ref_imgs_info["trans"]#2, 
    trans_src = src_imgs_info["trans"]#2, 
    trans = torch.stack([torch.stack([trans_src[idx], trans_ref[idx]], dim=0) \
        for idx in range(len(pano_ref))], dim=0)
    del pano_ref, pano_src
    return panos, rots, trans 
def merge_mv(input_ref, input_src, ref_imgs_info):
    # pano_src = F.interpolate(src_imgs_info["imgs"], (cfg["depth_height"], cfg["depth_width"]), mode='bilinear') #2, _, H, W
    # panos = []
    # for ref_idx in range(len(pano_ref)):#batch
    #     src_nn_ids = ref_imgs_info['nn_ids'][ref_idx]
    #     # torch.stack([])
    #     import ipdb;ipdb.set_trace()
    #     panos.append(torch.cat([pano_ref[ref_idx].unsqueeze(0), pano_src[src_nn_ids]], dim=0))
    # panos = torch.stack(panos, dim=0) 
    outputs = []
    for ref_idx in range(len(input_ref)):#batch
        src_nn_ids = ref_imgs_info['nn_ids'][ref_idx]
        # torch.stack([])
        # import ipdb;ipdb.set_trace()
        outputs.append(torch.cat([input_ref[ref_idx].unsqueeze(0), input_src[src_nn_ids]], dim=0))
    outputs = torch.stack(outputs, dim=0)
    return outputs


def construct_input_data_mv(cfg, ref_imgs_info, src_imgs_info):
    pano_ref = F.interpolate(ref_imgs_info["imgs"], (cfg["depth_height"], cfg["depth_width"]), mode='bilinear') #2, _, H, W
    pano_src = F.interpolate(src_imgs_info["imgs"], (cfg["depth_height"], cfg["depth_width"]), mode='bilinear') #2, _, H, W
    panos = merge_mv(pano_ref, pano_src, ref_imgs_info)
    # import ipdb;ipdb.set_trace()
    # panos = torch.stack([torch.stack([pano_src[idx], pano_ref[idx]], dim=0) \
    #     for idx in range(len(pano_ref))], dim=0)    
    
    rot_ref = ref_imgs_info["rots"]#2, 
    rot_src = src_imgs_info["rots"]#2, 
    rots = merge_mv(rot_ref, rot_src, ref_imgs_info)
    # rots = torch.stack([torch.stack([rot_src[idx], rot_ref[idx]], dim=0) \
    #     for idx in range(len(pano_ref))], dim=0)
    
    trans_ref = ref_imgs_info["trans"]#2, 
    trans_src = src_imgs_info["trans"]#2, 
    trans = merge_mv(trans_ref, trans_src, ref_imgs_info)
    # trans = torch.stack([torch.stack([trans_src[idx], trans_ref[idx]], dim=0) \
    #     for idx in range(len(pano_ref))], dim=0)
    # del pano_ref, pano_src
    return panos, rots, trans 

# def construct_mono_input_data(cfg, ref_imgs_info, src_imgs_info):
#     pano_ref = F.interpolate(ref_imgs_info["imgs"], (cfg["depth_height"], cfg["depth_width"]), mode='bilinear')#2, _, H, W
#     # pano_src = F.interpolate(src_imgs_info["imgs"], (cfg["depth_height"], cfg["depth_width"]), mode='bilinear') #2, _, H, W
#     # import ipdb;ipdb.set_trace()
#     # panos = torch.stack([torch.stack([pano_src[idx], pano_ref[idx]], dim=0) \
#     #     for idx in range(len(pano_ref))], dim=0)    
#     # del pano_ref, pano_src
#     return panos_ref
#todo?
def construct_cost_volume_with_src(args,
        ref_imgs_info, src_imgs_info, mvsnet, is_train):
    # ref_imgs = ref_imgs_info['imgs']#2(ref_num), 3, h, w
    # src_imgs = src_imgs_info['imgs']#2(ref_num), 3, h, w
    
   
    # ratio = 1.0
    # if resize:
    #     # ref_imgs = ref_imgs[:,:,:756,:1008] # 768, 1024
    #     if h == 768 and w == 1024:
    #         ref_imgs_ = F.interpolate(ref_imgs, (576, 768), mode='bilinear')
    #         src_imgs_ = F.interpolate(src_imgs, (576, 768), mode='bilinear')
    #         ratio = 576 / 768
    #     elif h == 800 and w == 800:
    #         ref_imgs_ = F.interpolate(ref_imgs, (640, 640), mode='bilinear')
    #         src_imgs_ = F.interpolate(src_imgs, (640, 640), mode='bilinear')
    #         ratio = 640 / 800
    #     else:
    # ref_imgs_ = ref_imgs
    # src_imgs_ = src_imgs
    ratio = 1.0

    with torch.no_grad():
        # nn_ids = ref_imgs_info['nn_ids']  # rfn,nn
        #todo?:construct_project_matrix, get_depth_vals
        # ref_prj = construct_project_matrix(0.25 * ratio, 0.25 * ratio, ref_imgs_info['Ks'], ref_imgs_info['poses'])
        # src_prj = construct_project_matrix(0.25 * ratio, 0.25 * ratio, src_imgs_info['Ks'], src_imgs_info['poses'])
        # depth_vals = get_depth_vals(ref_imgs_info['depth_range'], cost_volume_sn)  # rfn,dn
        # ref_imgs_imagenet = (ref_imgs_ - imagenet_mean) / imagenet_std
        # src_imgs_imagenet = (src_imgs_ - imagenet_mean) / imagenet_std
        mvsnet.eval()
        batch_num = 1 if not is_train else 2
        if not is_train:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # if args["debug"]:
        #     panos, rots, trans = construct_input_data(ref_imgs_info, src_imgs_info)
        #     # import ipdb;ipdb.set_trace()
        #     panos = panos.permute((0, 1, 3, 4, 2))
        #     ret_data = mvsnet.estimate_depth_using_cost_volume_v3_erp(panos, rots, trans, args["min_depth"], args["max_depth"])
        #     cost_reg = ret_data["cost_reg"]
        #     # import ipdb;ipdb.set_trace()
        #     cost_reg = cost_reg.permute((0, 3, 1, 2))

        #     depth = ret_data["depth"]
        #     depth = depth.permute((0, 3, 1, 2))


        # else:
            # try:
        #todo?
        #mvs_min_depth
        # import ipdb;ipdb.set_trace()#panos.shape[1]
        if args["seq_len"] > 3: #curr_idx:0
            panos, rots, trans = construct_input_data_mv(args, ref_imgs_info, src_imgs_info)
            panos = panos.permute((0, 1, 3, 4, 2))
            # we put the reference view in the first place:
            curr_idx = 0
            ret_data = mvsnet.estimate_depth_using_cost_volume_v3_erp_multiview(panos, rots, trans, curr_idx, args["mvs_min_depth"], args["mvs_max_depth"])
        else:
            panos, rots, trans = construct_input_data(args, ref_imgs_info, src_imgs_info)
            panos = panos.permute((0, 1, 3, 4, 2))
            ret_data = mvsnet.estimate_depth_using_cost_volume(panos, rots, trans, args["mvs_min_depth"], args["mvs_max_depth"])


        cost_reg = ret_data["cost_reg"]
        cost_reg = cost_reg.permute((0, 3, 1, 2))
        depth = ret_data["depth"]
        depth = torch.clamp_min(depth, 0.0)
        
        if args["uncert_tune"]:
            var = ret_data["var"]        
        #visualize
        # import cv2
        # depth_np = depth[0, :, :, 0].data.cpu().numpy()
        # rgb = ref_imgs_info["imgs"][0].permute((1, 2, 0)).data.cpu().numpy()
        # rgb = np.uint8(rgb*255)
        # d_min = np.min(depth_np)
        # d_max = np.max(depth_np)
        # d_scaled = np.uint8((depth_np - d_min)/(d_max - d_min)*255)

        # d_color = cv2.applyColorMap(d_scaled, cv2.COLORMAP_JET)
        
        # # import ipdb;ipdb.set_trace()
        # cv2.imwrite("d_color1.jpg", d_color)
        # cv2.imwrite("c_color1.jpg", rgb)
        
        # import ipdb;ipdb.set_trace()

        depth = depth.permute((0, 3, 1, 2))

            # except RuntimeError:
            #     # import ipdb; ipdb.set_trace()
            #     print("Error")
            #     exit
        cost_reg[torch.isnan(cost_reg)] = 0
        # if resize: cost_reg = F.interpolate(cost_reg, (h // 4, w // 4), mode='bilinear')
        cost_reg = F.softmax(cost_reg, 1)
    # depth = depth_regression(cost_reg, depth_vals)
    if args["uncert_tune"]:
        return cost_reg, depth, var
    else:
        return cost_reg, depth

def construct_monodepth_with_src(args,
        ref_imgs_info, src_imgs_info, dnet, is_train, e2c_mono, normalize_fn):
    ratio = 1.0
    with torch.no_grad():
        dnet.eval()
        batch_num = 1 if not is_train else 2
        if not is_train:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        if args["mono_net"]=="UniFuse":
            panos = F.interpolate(ref_imgs_info["imgs"], (args["mono_height"], args["mono_width"]), mode='bilinear')#2, _, H, 
            panos = panos.permute((0, 2, 3, 1))
            panos_cube = e2c_process(panos, e2c_mono)
            panos = panos.permute((0, 3, 1, 2))
            # import ipdb;ipdb.set_trace()
            # print('panos.shape:', panos.shape)
            panos_cube = panos_cube.permute((0, 3, 1, 2))
            erp_inputs, cube_inputs = normalize_input(panos, panos_cube, normalize_fn)
            # import ipdb;ipdb.set_trace()
            outputs = dnet(erp_inputs.cuda().cuda(), cube_inputs.cuda())
            depth = outputs["pred_depth"] 
            vis=False
            if vis:
                depth_np = depth.data.cpu().numpy()
                d_min = depth_np.min()
                d_max = depth_np.max()
                grey = np.uint8((depth_np - d_min)/(d_max - d_min)*255)
                grey_0 = grey[0, 0]
                d_rgb = cv2.applyColorMap(grey_0, cv2.COLORMAP_JET)
                rgb = np.uint8(255*ref_imgs_info["imgs"].permute((0, 2, 3, 1))[0].data.cpu().numpy())
                # cv2.imwrite("")
                cv2.imwrite("d_rgb.jpg", d_rgb)
                cv2.imwrite("rgb.jpg", rgb)
                import ipdb;ipdb.set_trace()
            depth = F.interpolate(depth, (args["depth_height"], args["depth_width"]), mode='bilinear')        
            depth = torch.clamp_min(depth, 0.0)
    
    return depth

def get_depth_vals(depth_range, dn):
    near = depth_range[:, 0]
    far = depth_range[:, 1]
    interval = (1/far - 1/near)/(dn-1) # rfn
    depth_vals = 1/(1/near[:,None] + torch.arange(0,dn-1,device=depth_range.device)[None,:]*interval[:,None]) # rfn,dn-1
    depth_vals = torch.cat([depth_vals,far[:,None]],1)
    return depth_vals # rfn,dn

# def homo_warping(src_fea, src_proj, ref_proj, depth_values):
#     # src_fea: [B, C, H, W]
#     # src_proj: [B, 4, 4]
#     # ref_proj: [B, 4, 4]
#     # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
#     # out: [B, C, Ndepth, H, W]
#     batch, channels = src_fea.shape[0], src_fea.shape[1]
#     num_depth = depth_values.shape[1]
#     height, width = src_fea.shape[2], src_fea.shape[3]

#     with torch.no_grad():
#         proj = torch.matmul(src_proj, torch.inverse(ref_proj))
#         rot = proj[:, :3, :3]  # [B,3,3]
#         trans = proj[:, :3, 3:4]  # [B,3,1]

#         y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
#                                torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
#         y, x = y.contiguous(), x.contiguous()
#         y, x = y.view(height * width), x.view(height * width)
#         xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
#         xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
#         rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
#         rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
#         proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
#         proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
#         proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
#         proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
#         proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
#         grid = proj_xy

#     warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear', padding_mode='zeros')

#     warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

#     return warped_src_fea

#todo?
class CostVolumeInitNet(nn.Module):
    default_cfg={
        'cost_volume_sn': 64,
        'patchsize': (128, 128),

    }
    def __init__(self,cfg):
        super().__init__()
        self.cfg={**self.default_cfg,**cfg}
        # note we do not train MVSNet here
        # self.mvsnet = MVSNet(ABN)
        args = self.cfg
        use_wrap_padding = self.cfg["use_wrap_padding"]
        # print('self.cfg:', self.cfg)
        #todo:args?(mvsnet_pretrained_path...)
        # import ipdb;ipdb.set_trace()
        # print("height, width:", self.cfg["height"], self.cfg["width"])
        # self.mvsnet = FullPipeline(args,
        #                  width=args["width"],
        #                  height=args["height"],
        #                  layers=5,
        #                  raster_resolution=args["width"],
        #                  depth_input_images=1,
        #                  depth_output_channels=1,
        #                  include_poseestimator=True,
        #                  verbose=args["verbose"],
        #                  input_uv=args["depth_input_uv"],
        #                  interpolation_mode=args["interpolation_mode"],
        #                  cost_volume=args["cost_volume"],
        #                  use_v_input=args["model_use_v_input"],
        #                  )
        self.depth_args = {**self.cfg}
        self.depth_args["min_depth"] = args["mvs_min_depth"]
        self.depth_args["max_depth"] = args["mvs_max_depth"]
        self.depth_args["width"] = args["depth_width"]
        self.depth_args["height"] = args["depth_height"]
        # import ipdb;ipdb.set_trace()
        if "wo_stereo" in args and args["wo_stereo"]:        
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            from select_mononet import select_mono
            self.depth_net = select_mono(args, mvsnet=True)      
            # import ipdb;ipdb.set_trace()
            self.depth_net.to(args["device"])
            if args["mono_uncert_tune"]:
                from network.omni_mvsnet.mono_uncert_wrapper import MonoUncertWrapper
                self.depth_net = MonoUncertWrapper(args, self.depth_net)
            self.depth_net = load_checkpoint(args["DNET_ckpt"], self.depth_net, "model_state_dict")#"dnet")
            for param in self.depth_net.parameters():
                param.requires_grad = False
            self.depth_net.eval()
            self.e2c_mono = Equirec2Cube(args["mono_height"], args["mono_width"], args["mono_height"] // 2)

        else:
            self.depth_net = FullPipeline(self.depth_args,
                                width=self.depth_args["width"],
                                height=self.depth_args["height"],
                                layers=5,
                                raster_resolution=self.depth_args["width"],
                                depth_input_images=1,
                                depth_output_channels=1,
                                include_poseestimator=True,
                                verbose=args["verbose"],
                                input_uv=args["depth_input_uv"],
                                interpolation_mode=args["interpolation_mode"],
                                cost_volume=args["cost_volume"],
                                use_v_input=args["model_use_v_input"],
                                ).to(args["device"])

            if args["uncert_tune"]:
                # load_mvs_model(self.mvs_net, args["mvs_checkpoints_dir"])            
                # for param in self.mvs_net.parameters():
                #   param.requires_grad = False
                # self.mvs_net.eval()

                from network.omni_mvsnet.uncert_wrapper import UncertWrapper
                self.depth_net = UncertWrapper(args, self.depth_net)

            if args["debug"]:
                pass
            else:
                # load_ckpt(self.mvsnet, args["mvsnet_pretrained_path"])            
                # import ipdb;ipdb.set_trace()
                
                self.depth_net.load_state_dict(torch.load(args["mvsnet_pretrained_path"], map_location=args["device"])['model_state_dict'])
                for para in self.depth_net.parameters():
                    para.requires_grad = False
                self.depth_net.eval()#

        # imagenet_mean = torch.from_numpy(np.asarray([0.485, 0.456, 0.406], np.float32)).cuda()[None, :, None, None]
        # imagenet_std = torch.from_numpy(np.asarray([0.229, 0.224, 0.225], np.float32)).cuda()[None, :, None, None]
        # self.register_buffer('imagenet_mean', imagenet_mean)
        # self.register_buffer('imagenet_std', imagenet_std)

        if "init_net_feature_type" in self.cfg and self.cfg["init_net_feature_type"] == "ERP+TP":
            npatches_dict = {3:10, 4:18, 5:26, 6:46}
            # def __init__(self, in_dim=3, layers=(2, 3, 6, 3), out_dim=32, inplanes=32, use_wrap_padding=False,
            #     fusion_type="cee", se_in_fusion=True,  nrows=4, npatches=18, patch_size=(128, 128), fov=(80, 80)                 

            self.res_net = ResUNetLight_ERP_TP(out_dim=32, use_wrap_padding=use_wrap_padding, \
                fusion_type=self.cfg["fusion"], se_in_fusion=self.cfg["se_in_fusion"],  nrows=self.cfg['nrows'], \
                 npatches=npatches_dict[self.cfg["nrows"]], patch_size=self.cfg["patchsize"], fov=self.cfg["fov"],
                 autoencoder=self.cfg["autoencoder"]
            )
        else:#ERP only
            self.res_net = ResUNetLight(self.cfg, out_dim=32, use_wrap_padding=use_wrap_padding)

        norm_layer = lambda dim: nn.InstanceNorm2d(dim, track_running_stats=False, affine=True)
        # volume_dim=64
        # if self.cfg["handle_distort_all"]:
        #     volume_dim+=1
        # self.volume_conv2d = nn.Sequential(
        #     conv3x3(volume_dim, 32, use_wrap_padding=use_wrap_padding),
        #     ResidualBlock(32, 32, norm_layer=norm_layer, use_wrap_padding=use_wrap_padding),
        #     conv1x1(32, 32, use_wrap_padding=use_wrap_padding),
        # )

        in_dim = 32+32
        depth_dim = 32
        self.depth_conv = nn.Sequential(
            conv3x3(1, depth_dim, use_wrap_padding=use_wrap_padding),
            ResidualBlock(depth_dim, depth_dim, norm_layer=norm_layer, use_wrap_padding=use_wrap_padding),
            conv1x1(depth_dim, depth_dim, use_wrap_padding=use_wrap_padding),
        )

        # in_dim+=depth_dim

        if self.cfg["handle_distort"]:
        #     # Polar Branch
        #     self.polarcoord = nn.Sequential(
        #         conv3x3(1, 32, use_wrap_padding=use_wrap_padding),
        #         ResidualBlock(32, 32, norm_layer=norm_layer, use_wrap_padding=use_wrap_padding),
        #         conv1x1(32, 32, use_wrap_padding=use_wrap_padding),
        #     )
        #     in_dim+=32

        #     # bs, _, fh, fw = volume_feats.shape
            height= self.cfg["depth_height"]
            width = self.cfg["depth_width"]
            batch_size = self.cfg["batch_size"]
            sin_phi = torch.arange(0, height//4, dtype=torch.float32).cuda()
            sin_phi = torch.sin((sin_phi + 0.5) * math.pi / (height//4))
            sin_phi = sin_phi.view(1, 1, height//4, 1).expand(batch_size, 1, height//4, width//4)
            self.sin_phi = sin_phi

        
        # if self.cfg["handle_distort_all"]:
        #     in_dim+=1
        self.out_conv = nn.Sequential(
            conv3x3(in_dim, 32, use_wrap_padding=use_wrap_padding),
            ResidualBlock(32, 32, norm_layer=norm_layer, use_wrap_padding=use_wrap_padding),
            conv1x1(32, 32, use_wrap_padding=use_wrap_padding),
        )



    def forward(self, ref_imgs_info, src_imgs_info, is_train):
        if "wo_stereo" in self.cfg and self.cfg["wo_stereo"]:
            depth = construct_monodepth_with_src(self.cfg, ref_imgs_info, src_imgs_info, self.depth_net, is_train, self.e2c_mono, self.normalize)
        else:
            if self.cfg["uncert_tune"]:
                cost_reg, depth, var = construct_cost_volume_with_src(self.cfg, ref_imgs_info, src_imgs_info, self.depth_net, is_train)
            else:
                cost_reg, depth = construct_cost_volume_with_src(self.cfg, ref_imgs_info, src_imgs_info, self.depth_net, is_train)
       
        ret = {
            "mvs_depth": depth,
        }
        # import ipdb;ipdb.set_trace()
        if self.cfg["uncert_tune"]:
            ret["mvs_uncert"]=var

        if "use_gt_depth_ray" in self.cfg and self.cfg["use_gt_depth_ray"]:
            # ret["mvs_depth"] = ref_imgs_info["depth"]
            depth = ref_imgs_info['depth']
        # if self.cfg["autoencoder"]:
        #     ref_feats, ray_ae_outputs = self.res_net(ref_imgs_info['imgs'])
        # else:
        # if self.cfg["autoencoder"]:
        #     ref_feats, ae_outputs = self.res_net(ref_imgs_info['imgs'])

        # else:

        
        ref_feats = self.res_net(F.interpolate(ref_imgs_info['imgs'], (self.cfg["depth_height"], self.cfg["depth_width"]), mode='bilinear'))
               
        # if self.cfg["handle_distort_all"]:
        #     bs, _, fh, fw = ref_feats.shape
        #     sin_phi = self.sin_phi.expand(bs, 1, fh, fw)
        # if self.cfg["handle_distort_all"]:
        #     volume_feats = self.volume_conv2d(torch.cat([cost_reg, sin_phi], dim=1))
        # else:    
        #     volume_feats = self.volume_conv2d(cost_reg)
        
        
        depth = extract_depth_for_init_impl(self.cfg,  depth)
        depth = nn.functional.interpolate(
            depth,
            scale_factor=0.25,
            mode="bilinear",
            align_corners=False
        )#.permute((0, 2, 3, 1))

        depth_feats = self.depth_conv(depth)
        # import ipdb;ipdb.set_trace()

        
        # volume_feats = torch.cat([volume_feats, depth_feats],1)

        # if self.cfg["handle_distort_all"]:
        #     volume_feats = torch.cat([volume_feats, sin_phi], dim=1)
        #     polar_feats=self.polarcoord(self.sin_phi.expand(bs, 1, fh, fw).clone())
        #     volume_feats = torch.cat([volume_feats, polar_feats], dim=1)

        ray_feats = self.out_conv(torch.cat([ref_feats, depth_feats], 1))

        ret["ray_feats"] = ray_feats
        if "use_empty_depth_ray" in self.cfg and self.cfg["use_empty_depth_ray"]:
            bs, dim, fh, fw = ray_feats.shape
            # fh, fw = 64, 128 #self.cfg['ray_feats_res']
            # dim = 32 #self.cfg['ray_feats_dim']
            # ref_num = len(self.ref_ids)
            # for k in range(ref_num):
            ret["ray_feats"] = nn.Parameter(torch.randn(bs,dim,fh,fw)).cuda()
        return ret

name2init_net={
    # 'depth': DepthInitNet,
    'cost_volume': CostVolumeInitNet,
}