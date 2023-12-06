import torch
from network.ops import interpolate_feats
from network.ray_utils import get_sphere_ray_directions, get_sphere_rays, cartesian_2_equi
from network.sample_utils import sample_3sigma, perturb_z_vals
# #?
# def coords2rays(coords, poses, Ks):
#     """
#     :param coords:   [rfn,rn,2]
#     :param poses:    [rfn,3,4]
#     :param Ks:       [rfn,3,3]
#     :return:
#         ref_rays:
#             centers:    [rfn,rn,3]
#             directions: [rfn,rn,3]
#     """
#     rot = poses[:, :, :3].unsqueeze(1).permute(0, 1, 3, 2)  # rfn,1,3,3
#     trans = -rot @ poses[:, :, 3:].unsqueeze(1)  # rfn,1,3,1

#     rfn, rn, _ = coords.shape

#     centers = trans.repeat(1, rn, 1, 1).squeeze(-1)  # rfn,rn,3

#     coords = torch.cat([coords, torch.ones([rfn, rn, 1], dtype=torch.float32, device=coords.device)], 2)  # rfn,rn,3

#     Ks_inv = torch.inverse(Ks).unsqueeze(1)
#     cam_xyz = Ks_inv @ coords.unsqueeze(3)
#     cam_xyz = rot @ cam_xyz + trans

#     directions = cam_xyz.squeeze(3) - centers

    
#     # directions = directions / torch.clamp(torch.norm(directions, dim=2, keepdim=True), min=1e-4)
    
    
#     return centers, directions

def coords2rays(coords, poses, Ks):
    """
    :param coords:   [rfn,rn,2]
    :param poses:    [rfn,3,4]
    :param Ks:       [rfn,3,3]
    :return:
        ref_rays:
            centers:    [rfn,rn,3]
            directions: [rfn,rn,3]
    """
    rot = poses[:, :, :3].unsqueeze(1).permute(0, 1, 3, 2)  # rfn,1,3,3
    trans = -rot @ poses[:, :, 3:].unsqueeze(1)  # rfn,1,3,1

    rfn, rn, _ = coords.shape
    centers = trans.repeat(1, rn, 1, 1).squeeze(-1)  # rfn,rn,3
    coords = torch.cat([coords, torch.ones([rfn, rn, 1], dtype=torch.float32, device=coords.device)], 2)  # rfn,rn,3
    Ks_inv = torch.inverse(Ks).unsqueeze(1)
    cam_xyz = Ks_inv @ coords.unsqueeze(3)
    
    cam_xyz = rot @ cam_xyz + trans # c2w

    directions = cam_xyz.squeeze(3) - centers
    # directions = directions / torch.clamp(torch.norm(directions, dim=2, keepdim=True), min=1e-4)
    return centers, directions

def depth2points_perspec(que_imgs_info, que_depth):
    """
    :param que_imgs_info:
    :param que_depth:       qn,rn,dn
    :return:
    """
    cneters, directions = coords2rays(que_imgs_info['coords'],que_imgs_info['poses'],que_imgs_info['Ks']) # centers, directions qn,rn,3
    qn, rn, _ = cneters.shape
    que_pts = cneters.unsqueeze(2) + directions.unsqueeze(2) * que_depth.unsqueeze(3) # qn,rn,dn,3
    qn, rn, dn, _ = que_pts.shape
    que_dir = -directions / torch.norm(directions, dim=2, keepdim=True)  # qn,rn,3
    que_dir = que_dir.unsqueeze(2).repeat(1, 1, dn, 1)
    return que_pts, que_dir # qn,rn,dn,3

def depth2points_spherical(que_imgs_info, que_depth, spt_utils):
    """
    :param que_imgs_info:
    :param que_depth:       qn,rn,dn
    :return:
    """
    #todo?(c2w)
    c2w=que_imgs_info["c2w"]
    device=c2w.device
    directions_all = get_sphere_ray_directions(
            spt_utils)  # (h, w, 3)
    h, w = directions_all.shape[:2]
    qn, rn, dn = que_depth.shape
    assert c2w.shape[0] == 1, "que_imgs_info c2w.shape[0]=1"
    c2w = c2w.squeeze()
    rays_o, rays_d = get_sphere_rays(
                directions_all.float().to(device), c2w)  # both (h*w, 3)
    # import ipdb;ipdb.set_trace()    
    #??
    #??
    centers = rays_o.view(h, w, 3)[que_imgs_info['coords'][:, :, 1].long(), que_imgs_info['coords'][:, :, 0].long(), :]
    directions = rays_d.view(h, w, 3)[que_imgs_info['coords'][:, :, 1].long(), que_imgs_info['coords'][:, :, 0].long(), :]
    assert centers.shape[0] == qn and centers.shape[1] == rn, "centers.shape error"
    qn, rn, _ = centers.shape
    que_pts = centers.unsqueeze(2) + directions.unsqueeze(2) * que_depth.unsqueeze(3) # qn,rn,dn,3
    
    qn, rn, dn, _ = que_pts.shape
    #todo?
    que_dir = -directions / torch.norm(directions, dim=2, keepdim=True)  # qn,rn,3 #?    
    que_dir = que_dir.unsqueeze(2).repeat(1, 1, dn, 1)
    return que_pts, que_dir # qn,rn,dn,3



def depth2dists(depth):
    device = depth.device
    dists = depth[...,1:]-depth[...,:-1]
    return torch.cat([dists, torch.full([*depth.shape[:-1], 1], 1e6, dtype=torch.float32, device=device)], -1)

def depth2inv_dists(depth, depth_range):
    # import ipdb;ipdb.set_trace()
    near, far = -1 / depth_range[:, 0], -1 / depth_range[:, 1]
    near, far = near[:, None, None], far[:, None, None]
    depth_inv = -1 / depth  # qn,rn,dn
    depth_inv = (depth_inv - near) / (far - near)
    dists = depth2dists(depth_inv)  # qn,rn,dn
    return dists

#todo? border_type(ERP?)

def interpolate_feature_map(ray_feats, coords, h, w, border_type='border'):
    """
    :param ray_feats:       rfn,f,h,w
    :param coords:          rfn,pn,2
    :param mask:            rfn,pn
    :param h:
    :param w:
    :param border_type:
    :return:
    """
    fh, fw = ray_feats.shape[-2:]
    if fh == h and fw == w:
        cur_ray_feats = interpolate_feats(ray_feats, coords, h, w, border_type, True)  # rfn,pn,f
    else:
        cur_ray_feats = interpolate_feats(ray_feats, coords, h, w, border_type, False)  # rfn,pn,f

    # cur_ray_feats = cur_ray_feats * mask.float().unsqueeze(-1) # rfn,pn,f
    return cur_ray_feats

def alpha_values2hit_prob(alpha_values):
    """
    :param alpha_values: qn,rn,dn
    :return: qn,rn,dn
    """
    no_hit_density = torch.cat([torch.ones((*alpha_values.shape[:-1], 1))
                               .to(alpha_values.device), 1. - alpha_values + 1e-10], -1)  # rn,k+1
    hit_prob = alpha_values * torch.cumprod(no_hit_density, -1)[..., :-1]  # [n,k]
    return hit_prob


#todo?
#3d -> 2d
def project_points_coords(pts, Rt, spt_utils):
    """
    :param pts:  [pn,3]
    :param Rt:   [rfn,3,4]#pose
    :return:
        coords:         [rfn,pn,2]
        invalid_mask:   [rfn,pn]
    """
    pn = pts.shape[0]
    hpts = torch.cat([pts,torch.ones([pn,1],device=pts.device,dtype=torch.float32)],1)#pn, 4

    srn = Rt.shape[0]
    Rt # rfn,3,4

    last_row = torch.zeros([srn,1,4],device=pts.device,dtype=torch.float32)
    last_row[:,:,3] = 1.0
    H = torch.cat([Rt,last_row],1) # rfn,4,4
    pts_cam = H[:,None,:,:] @ hpts[None,:,:,None]#rfn, 1, 4, 4 @ 1, pn, 4, 1->rfn, pn, 4 , 1
    # import ipdb;ipdb.set_trace()
    pts_cam = pts_cam[:,:,:3,0]#rfn, pn, 3
    depth, pixel_coords = cartesian_2_equi(spt_utils, pts_cam)
    # if torch.isnan(depth).any() or torch.isnan(pixel_coords).any():
    #     import ipdb;ipdb.set_trace()

    # depth: 2, 131072
    #bug? pixel_coords[:, :, 1].long().unique() #max() very small256
    # import ipdb;ipdb.set_trace()
    # print('depth.shape:', depth.shape) 
    # print("pixel_coords[:")
    # depth = pts_cam[:,:,2:]
    # invalid_mask = torch.abs(depth)<1e-4
    # depth[invalid_mask] = 1e-3
    # pts_2d = pts_cam[:,:,:2]/depth
    return pixel_coords, depth


#todo?
def project_points_directions(poses,points):
    """
    :param poses:       rfn,3,4
    :param points:      pn,3
    :return: rfn,pn,3
    """
    #rfn, 3, 3 -> ref, 3, 3
    #cam_pts = -R^T@T
    cam_pts = -poses[:, :, :3].permute(0, 2, 1) @ poses[:, :, 3:]  # rfn,3,1

    # 
    dir = points.unsqueeze(0) - cam_pts.permute(0, 2, 1)  # [1,pn,3] - [rfn,1,3] -> rfn,pn,3
    dir = -dir / torch.clamp_min(torch.norm(dir, dim=2, keepdim=True), min=1e-5)  # rfn,pn,3
    # dir = dir / torch.clamp_min(torch.norm(dir, dim=2, keepdim=True), min=1e-5)  # rfn,pn,3
    return dir



#todo?
def project_points_ref_views(ref_imgs_info, que_points, spt_utils):
    """
    :param ref_imgs_info:
    :param que_points:      pn,3
    :return:
    """
    prj_pts, prj_depth = project_points_coords(
        que_points, ref_imgs_info['w2c'], spt_utils) # rfn,pn,2
    
    h, w=ref_imgs_info['imgs'].shape[-2:]
    # import ipdb;ipdb.set_trace()
    # prj_img_invalid_mask = (prj_pts[..., 0] < -0.5) | (prj_pts[..., 0] >= w - 0.5) | \
    #                        (prj_pts[..., 1] < -0.5) | (prj_pts[..., 1] >= h - 0.5)
    # valid_mask = prj_valid_mask & (~prj_img_invalid_mask)
    prj_dir = project_points_directions(ref_imgs_info['w2c'], que_points) # rfn,pn,3

    return prj_dir, prj_pts, prj_depth#, valid_mask


#todo?
def project_points_dict(ref_imgs_info, que_pts, spt_utils):
    # project all points
    qn, rn, dn, _ = que_pts.shape
    prj_dir, prj_pts, prj_depth = project_points_ref_views(ref_imgs_info, que_pts.reshape([qn * rn * dn, 3]), spt_utils)

    # print("prj_pts:", prj_pts)
    # print("y, prj_pts[0, :, 1].long().unique():",prj_pts[0, :, 1].long().unique())

    rfn, _, h, w = ref_imgs_info['imgs'].shape
    prj_ray_feats = interpolate_feature_map(ref_imgs_info['ray_feats'], prj_pts, h, w)

    prj_rgb = interpolate_feature_map(ref_imgs_info['imgs'], prj_pts, h, w)
    # import ipdb;ipdb.set_trace()
    #prj_dir: 2, 1, 2048, 64, -1
    # pts, rgb
    # todo: depth, ray_feats, dir
    prj_dict = {'dir':prj_dir, 'pts':prj_pts, 'depth':prj_depth, 'ray_feats':prj_ray_feats, 'rgb':prj_rgb}
    if torch.isnan(prj_ray_feats).any():
        import ipdb;ipdb.set_trace()
    # ?
    # post process
    for k, v in prj_dict.items():
        prj_dict[k]=v.reshape(rfn,qn,rn,dn,-1) 
    return prj_dict

#todo?
def project_points_dict_diner(ref_imgs_info, diner_que_pts, spt_utils, include_norm=False):
    # project all points
    qn, rn, dn, _ = diner_que_pts.shape
    prj_dir, prj_pts, prj_depth = project_points_ref_views(ref_imgs_info, diner_que_pts.reshape([qn * rn * dn, 3]), spt_utils)

    rfn, _, h, w = ref_imgs_info['imgs'].shape #
    # prj_ray_feats = interpolate_feature_map(ref_imgs_info['ray_feats'], prj_pts, h, w) #
    # prj_rgb = interpolate_feature_map(ref_imgs_info['imgs'], prj_pts, h, w)
    # import ipdb;ipdb.set_trace()
    prj_ref_depth = interpolate_feature_map(ref_imgs_info['mvs_depth'], prj_pts, h, w)
    prj_ref_uncert = interpolate_feature_map(ref_imgs_info['mvs_uncert'], prj_pts, h, w)

    if include_norm:
        # import ipdb;ipdb.set_trace()
        prj_ref_normal = interpolate_feature_map(ref_imgs_info['mvs_normal'], prj_pts, h, w)


    #prj_dir: 2, 1, 2048, 64, -1
    # pts, rgb
    # todo: depth, ray_feats, dir
    prj_dict = {'ref_mvs_depths':prj_ref_depth, 'ref_mvs_uncert': prj_ref_uncert, 'pts':prj_pts, 'depth':prj_depth}
    if include_norm:
        prj_dict['ref_mvs_normal'] = prj_ref_normal
        # import ipdb;ipdb.set


    # ?
    # post process
    for k, v in prj_dict.items():
        prj_dict[k]=v.reshape(rfn,qn,rn,dn,-1) 
    return prj_dict

def sample_depth(args, coords, sample_num, random_sample, use_disp=True):
    """
    :param depth_range: qn,2
    :param sample_num:
    :param random_sample:
    :return:
    """
    qn, rn, _ = coords.shape
    device = coords.device
    
    near, far = torch.ones((qn, )).to(device)*args["min_depth"], torch.ones((qn, )).to(device)*args["max_depth"]#depth_range[:,0], depth_range[:,1] # qn,2
    if not use_disp: #:args["use_disp"]:
        dn = sample_num
        assert(dn>2)
        interval = (far - near) / (dn - 1)  # qn
        val = torch.arange(1, dn - 1, dtype=torch.float32, device=device)[None, None, :]

        # import ipdb;ipdb.set_trace()
        if random_sample:
            val = val + (torch.rand(qn, rn, dn-2, dtype=torch.float32, device=device) - 0.5) * 0.999
        else:
            val = val + torch.zeros(qn, rn, dn-2, dtype=torch.float32, device=device)
        
        ticks = interval[:, None, None] * val
        diff = (far - near)
        ticks = torch.cat([torch.zeros(qn,rn,1,dtype=torch.float32,device=device),ticks,diff[:,None,None].repeat(1,rn,1)],-1)  

        que_depth = near[:, None, None] + ticks  # qn, dn,
        que_dists = torch.cat([que_depth[...,1:],torch.full([*que_depth.shape[:-1],1],1e6,dtype=torch.float32,device=device)],-1) - que_depth

        # pass
    else:

        dn = sample_num
        assert(dn>2)
        interval = (1 / far - 1 / near) / (dn - 1)  # qn
        val = torch.arange(1, dn - 1, dtype=torch.float32, device=device)[None, None, :]
        if random_sample:
            val = val + (torch.rand(qn, rn, dn-2, dtype=torch.float32, device=device) - 0.5) * 0.999
        else:
            val = val + torch.zeros(qn, rn, dn-2, dtype=torch.float32, device=device)
        ticks = interval[:, None, None] * val

        diff = (1 / far - 1 / near)
        ticks = torch.cat([torch.zeros(qn,rn,1,dtype=torch.float32,device=device),ticks,diff[:,None,None].repeat(1,rn,1)],-1)
        que_depth = 1 / (1 / near[:, None, None] + ticks)  # qn, dn,
        que_dists = torch.cat([que_depth[...,1:],torch.full([*que_depth.shape[:-1],1],1e6,dtype=torch.float32,device=device)],-1) - que_depth
    return que_depth, que_dists # qn, rn, dn
#todo

# precomputed_z_samples, depth_range, perturb_z?, pytest
# sample_num:64

# self.cfg, que_imgs_info['coords'], self.cfg['depth_sample_num'], False
def sample_depth_with_guidance(args, coords, sample_num, random_sample, depth_range=None, precomputed_z_samples=None, pytest=False): 
    
    # N_samples_half = N_samples // 2 
    # N_samples_half = 
    qn, rn, _ = coords.shape
    near = args["min_depth"]
    far = args["max_depth"]
    # import ipdb;ipdb.set_trace()
    # depth_range = depth_range.squeeze(0)


    if random_sample:
        perturb = 1.0
    else:
        perturb = 0.0

    # if precomputed_z_samples is not None:
    #     # compute a lower bound for the sampling standard deviation as the maximal distance between samples
    #     lower_bound = precomputed_z_samples[-1] - precomputed_z_samples[-2] #

    # train time: use precomputed samples along the whole ray and additionally sample around the depth
    if depth_range is not None:
        depth_range = depth_range.squeeze(0)
        valid_depth = depth_range[..., 0] >= near
        invalid_depth = valid_depth.logical_not()
        # import ipdb;ipdb.set_trace()
        # do a forward pass for the precomputed first half of samples
        # z_vals = precomputed_z_samples.unsqueeze(0).expand((rn, sample_num//2))
        z_vals, _ = sample_depth(args, coords, sample_num//2, random_sample)
        # import ipdb;ipdb.set_trace()
        z_vals = z_vals.squeeze(0)

        if perturb > 0.:
            z_vals = perturb_z_vals(z_vals, pytest)
        # pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        # raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
        z_vals_2 = torch.empty_like(z_vals)
        # sample around the predicted depth from the first half of samples, if the input depth is invalid
        # z_vals_2[invalid_depth] = compute_samples_around_depth(raw.detach()[invalid_depth], z_vals[invalid_depth], rays_d[invalid_depth], N_samples_half, perturb, lower_bound, near[0, 0], far[0, 0])
        # sample with in 3 sigma of the input depth, if it is valid
        # import ipdb;ipdb.set_trace()
        # import ipdb;ipdb.set_trace()
        z_vals_2[valid_depth] = sample_3sigma(depth_range[valid_depth, 1], depth_range[valid_depth, 2], sample_num//2, perturb == 0., near, far)
        # import ipdb;ipdb.set_trace()
        z_vals_all, _ = torch.sort(torch.cat([z_vals, z_vals_2], -1), -1)

        #1. mix the basic depth samples
        #2. directly use the normal distribution
        #3. wo
        return z_vals_all.unsqueeze(0), _

        # return forward_with_additonal_samples(z_vals, raw, z_vals_2, rays_o, rays_d, viewdirs, embedded_cam, network_fn, network_query_fn, raw_noise_std, pytest)
    # test time: use precomputed samples along the whole ray and additionally sample around the predicted depth from the first half of samples
    elif precomputed_z_samples is not None:
        z_vals = precomputed_z_samples.unsqueeze(0).expand((rn, sample_num))
        # pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        # raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
        # z_vals_2 = compute_samples_around_depth(raw, z_vals, rays_d, N_samples_half, perturb, lower_bound, near[0, 0], far[0, 0])

        # return forward_with_additonal_samples(z_vals, raw, z_vals_2, rays_o, rays_d, viewdirs, embedded_cam, network_fn, network_query_fn, raw_noise_std, pytest)
        return z_vals.unsqueeze(0), _
    else:
        return sample_depth(args, coords, sample_num, random_sample)

# def args, coords, sample_num, random_sample


def sample_fine_depth(args, depth, hit_prob, depth_range, sample_num, random_sample, inv_mode=True):
    """
    :param depth:       qn,rn,dn
    :param hit_prob:    qn,rn,dn
    :param depth_range: qn,2
    :param sample_num:
    :param random_sample:
    :param inv_mode:
    :return: qn,rn,dn
    """
    if not args["use_disp"]:
        inv_mode=False

    if inv_mode:
        near, far = depth_range[0,0], depth_range[0,1]
        near, far = -1/near, -1/far
        depth_inv = -1 / depth  # qn,rn,dn
        depth_inv = (depth_inv - near) / (far - near)
        depth = depth_inv

    depth_center = (depth[...,1:] + depth[...,:-1])/2
    depth_center = torch.cat([depth[...,0:1],depth_center,depth[...,-1:]],-1) # rfn,pn,dn+1
    fdn = sample_num
    # Get pdf
    hit_prob = hit_prob + 1e-5  # prevent nans
    pdf = hit_prob / torch.sum(hit_prob, -1, keepdim=True) # rfn,pn,dn-1
    cdf = torch.cumsum(pdf, -1) # rfn,pn,dn-1
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # rfn,pn,dn

    # Take uniform samples
    if not random_sample:
        interval = 1 / fdn
        u = 0.5*interval+torch.arange(fdn)*interval
        # u = torch.linspace(0., 1., steps=fdn)
        u = u.expand(list(cdf.shape[:-1]) + [fdn]) # rfn,pn,fdn
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [fdn])

    # Invert CDF
    device = pdf.device
    u = u.to(device).contiguous() # rfn,pn,fdn
    inds = torch.searchsorted(cdf, u, right=True)                       # rfn,pn,fdn
    below = torch.max(torch.zeros_like(inds-1), inds-1)                 # rfn,pn,fdn
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)  # rfn,pn,fdn
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)   # rfn,pn,fdn,2

    matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)    # rfn,pn,fdn,2
    bins_g = torch.gather(depth_center.unsqueeze(-2).expand(matched_shape), -1, inds_g) # rfn,pn,fdn,2

    denom = (cdf_g[...,1]-cdf_g[...,0]) # rfn,pn,fdn
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    fine_depth = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    if inv_mode:
        near, far = depth_range[0,0], depth_range[0,1]
        near, far = -1/near, -1/far
        fine_depth = fine_depth * (far - near) + near
        fine_depth = -1/fine_depth
    return fine_depth


