import torch
import numpy as np
from torch.special import erf

def weighted_mean_n_std(x: torch.Tensor, weights: torch.Tensor, dim: int, keepdims=False):
    weights_normed = weights / weights.sum(dim=dim, keepdims=True)
    mean = (x * weights_normed).sum(dim=dim, keepdims=True)
    std = ((x - mean).pow(2) * weights_normed).sum(dim=dim, keepdims=True).sqrt()

    if not keepdims:
        mean = mean.squeeze(dim)
        std = std.squeeze(dim)
    return mean, std
# def sample_coarse(rays, n_coarse=None):
#     """
#     Stratified sampling. Note this is different from original NeRF slightly.
#     :param rays ray [origins (3), directions (3), near (1), far (1)] (SB, B, 8)
#     :return (B, Kc)
#     """
#     n_coarse = n_coarse if n_coarse else self.n_coarse

#     ray_shape = rays.shape
#     rays = rays.view(-1, 8)

#     device = rays.device
#     near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

#     step = 1.0 / n_coarse
#     B = rays.shape[0]
#     z_steps = torch.linspace(0, 1 - step, n_coarse, device=device)  # (Kc)
#     z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
#     z_steps += torch.rand_like(z_steps) * step

#     # Use linear sampling in depth space
#     z_samples = near * (1 - z_steps) + far * z_steps  # (B, Kc)
#     z_samples = z_samples.view(*ray_shape[:-1], n_coarse)

#     return z_samples


# 1. update normal calculation
# 1.5 fill up zero samples
# 2. try more combinations: diner+disp diner+uniform, diner only

@torch.no_grad()
def sample_depthguided(cfg, ref_imgs_info, prj_depth_info_dict, que_depth, que_dir, n_samples, n_candidates, n_gaussian, depth_diff_max=0.05, include_norm=False, var=True):

# def sample_depthguided(self, rays, model, n_samples, n_candidates,
#                            depth_diff_max=0.05, n_gaussian=None):
    """
    Ray sampling guided by reference depth maps
    :param rays: ray [origins (3), directions (3), near (1), far (1)] (batch_size, n_rays, 8)
    :param model: nerf model (instance of src.models.pixelnerf.PixelNeRF)
    :param n_samples: total number of samples per ray that NeRF is evaluated on
    :param n_candidates: number of candidate samples. Are shortlisted according to
                            surface likelihoods given by depth-predictions
    :param: depth_diff_max: maximal difference in sample point depth and predicted depth
                            if difference exceeds that value -> candidate is discarded
    :param: n_gaussian: number of gaussian samples drawn according to occlusion aware surface
                        likelihoods (replace shortlisted samples with smallest surface likelihoods)
    :return (batch_size, n_rays, n_samples) ... z values of the samples
    """

    # n_gaussian = n_gaussian if n_gaussian is not None else self.n_gaussian

    # # reducing ray nr (for debugging only!)
    # rays = rays[:, :10].clone()
    # print("WARNING: REDUCING RAY COUNT")

    assert n_samples >= n_gaussian
    # import ipdb;ipdb.set_trace()

    # SB, NV, _, _ = model.poses.shape

    # device = rays.device
    # NR = rays.shape[1]
    # z_samples = self.sample_coarse(rays, n_coarse=n_candidates)  # SB, NR, n_candidates
    mu = prj_depth_info_dict['ref_mvs_depths'].squeeze(-1)
    uncert = prj_depth_info_dict['ref_mvs_uncert'].squeeze(-1)
    prj_depth = prj_depth_info_dict['depth'].squeeze(-1) #?
    if include_norm: #
        prj_norm = prj_depth_info_dict['ref_mvs_normal'].squeeze(-1) #?
    if "diner_sigma" in cfg and cfg["diner_sigma"]>0:
        # import ipdb;ipdb.set_trace()
        sigma = torch.ones_like(mu)*cfg["diner_sigma"] #0.5
    else:
        if var:
            sigma = torch.sqrt(uncert)
        else:
            sigma = uncert




    


    # xyz = rays[..., None, :3] + z_samples.unsqueeze(-1) * rays[..., None, 3:6]

    # # Transform query points into the camera spaces of the input views
    # xyz = xyz.reshape(SB, -1, 3).unsqueeze(1).expand(-1, NV, -1, -1)  # (SB, NV, B, 3)
    # xyz_rot = torch.matmul(model.poses[:, :, :3, :3], xyz.transpose(-2, -1)).transpose(-2, -1)
    # xyz = xyz_rot + model.poses[:, :, :3, -1].unsqueeze(-2)  # (SB, NV, B, 3)
    # raydirs = rays[..., 3:6].unsqueeze(1).expand(SB, NV, NR, 3)  # SB, NV, NR, 3
    # raydirs_cam = (model.poses[:, :, :3, :3] @ raydirs.transpose(-2, -1)).transpose(-2, -1)  # SB, NV, NR, 3
    # import ipdb;ipdb.set_trace()
    #que_dir: 1, 512, 1000, 3 #B, N_ray, N_samples, 3
    #2, 3, 3->1, 2, 1, 3, 3
    #       ->1, 1, 512, 3, 1000
    #       ->1, 2, 512, 3, 1000 -> 1, 2, 512, 1000, 3
    negative=True
    if negative:
        que_dir = -que_dir

    que_dir_cam = (ref_imgs_info["w2c"][..., :3, :3].unsqueeze(1).unsqueeze(0) @ que_dir.transpose(-2, -1).unsqueeze(1)).transpose(-2, -1)

    # pointdirs_cam = raydirs_cam.repeat_interleave(n_candidates, dim=-2)  # (SB, NV, B, 3)

    # sample depth and normal maps
    # uv = xyz[..., :2] / xyz[..., 2:]  # (SB, NV, B, 2)
    # uv *= model.focal.unsqueeze(-2)
    # uv += model.c.unsqueeze(-2)
    # uv = uv / model.image_shape * 2 - 1  # assumes outer edges of pixels correspond to uv coordinates -1 / 1
    # ref_depth = model.encoder.index_depth(uv)  # SB, NV, 1, B
    # ref_depth_std = model.encoder.index_depth_std(uv)  # SB, NV, 1, B
    # ref_normal = model.encoder.index_normal(uv)  # SB, NV, 3, B
    # ref_z = xyz[..., 2:].permute(0, 1, 3, 2)  # SB, NV, 1, B


    # step_size = step_size.repeat_interleave(n_candidates, dim=1).view(SB, 1, 1, NR * n_candidates)
    # step_size = step_size.expand_as(ref_depth)  # SB, NV, 1, B
    
    # 2, 1, 512, 1000
    step_size = torch.ones_like(mu)*(cfg["max_depth"] - cfg["min_depth"]) / n_candidates  # SB, NR

    # step_size = step_size.expand_as(mu)

    # determining sample point likelihoods
    # que_dir_cam:1, 2, 512, 1000, 3
    # prj_norm:   2, 1, 512, 1000, 3
    #             1, 2, 512, 1000, 1
    cosdist_ray_normal = (que_dir_cam.transpose(0, 1) * prj_norm).sum(dim=-1, keepdim=False)  # 
    # SB, NV, 1, B
    
    pt_likelihood = torch.zeros_like(mu)  # NV, 1, N_ray, n_candidates
    if include_norm:
        cosdist_ray_normal_mask = cosdist_ray_normal <= 0 #debug
    else:
        cosdict_ray_normal_mask = torch.ones_like(cosdist_ray_normal)
    # import ipdb;ipdb.set_trace()

    depth_dist_mask = (mu - prj_depth).abs() < depth_diff_max
    # bg_mask = sigma != 0
    mask = depth_dist_mask & cosdist_ray_normal_mask  # SB, NV, 1, B

    pt_likelihood[mask] = 0.5 * (
            erf((prj_depth[mask] + step_size[mask] / 2 - mu[mask]) / (sigma[mask] * np.sqrt(2))) -
            erf((prj_depth[mask] - step_size[mask] / 2 - mu[mask]) / (sigma[mask] * np.sqrt(2)))
    ).abs()



    # import ipdb;ipdb.set_trace()
    #2, 1, 512, 1000
    pt_likelihood = torch.max(pt_likelihood, dim=0).values.squeeze(1).squeeze(0)  # SB, B
    # pt_likelihood = pt_likelihood.reshape(*rays.shape[:2], -1)  # SB, N_rays, N_pointsperray
    opaque_pt_likelihood = pt_likelihood.clone()
    opaque_pt_likelihood[..., 1:] *= torch.cumprod(1. - pt_likelihood, dim=-1)[..., :-1]

    # # visualize normals on reference images
    # import matplotlib.pyplot as plt
    # for i in range(NV):
    #     # 3D plot
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection="3d")
    #     cosdist_ray_normal_mask_ = cosdist_ray_normal_mask[0, i, 0]  # B
    #     s = .01
    #     ax.quiver(*xyz[0, i].cpu().unbind(dim=-1), *(pointdirs_cam * s)[0, i].cpu().unbind(dim=-1))
    #     ax.quiver(*xyz[0, i].cpu().unbind(dim=-1), *(ref_normal * s)[0, i].transpose(-2, -1).cpu().unbind(dim=-1),
    #               edgecolor="orange")
    #     ax.scatter(*xyz[0, i][cosdist_ray_normal_mask_].cpu().unbind(dim=-1), s=10.)
    #     ax.scatter(*xyz[0, i][~cosdist_ray_normal_mask_].cpu().unbind(dim=-1), s=10., c="red")
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("z")
    #     ax.set_xlim((-.5, .5))
    #     ax.set_ylim((-.5, .5))
    #     ax.set_zlim((1., 2.))
    #     fig.suptitle(str(i))
    #
    #     # 2D plot
    #     fig, ax = plt.subplots()
    #     ax.imshow(model.encoder.normals[0, i].permute(1, 2, 0).cpu() * .5 + .5)
    #     point_coords = ((uv + 1) / 2 * model.image_shape)[0, i]
    #     ax.quiver(*point_coords.cpu().unbind(dim=-1), *ref_normal[0, i, :2].transpose(-2, -1).cpu().unbind(dim=-1),
    #               angles="xy", scale_units='xy', scale=0.1, color="orange")
    #     ax.quiver(*point_coords.cpu().unbind(dim=-1), *pointdirs_cam[0, i, :, :2].cpu().unbind(dim=-1),
    #               angles="xy", scale_units='xy', scale=0.1, color="blue")
    #     ax.scatter(*point_coords[cosdist_ray_normal_mask_].cpu().unbind(dim=-1), s=5.)
    #     ax.scatter(*point_coords[~cosdist_ray_normal_mask_].cpu().unbind(dim=-1), s=5., c="red")
    #     ax.set_xlim((0, 256))
    #     ax.set_ylim((0, 256))
    #     ax.invert_yaxis()
    #     fig.suptitle(str(i))
    # plt.show()

    # determining sampling points
    selected_pts_idcs = pt_likelihood.argsort(dim=-1, descending=True)[..., :n_samples]  # SB, N_rays, n_depthsmpls

    # SB_helper = torch.arange(SB).view(-1, 1, 1).expand_as(selected_pts_idcs)
    # ray_helper = torch.arange(NR).view(1, -1, 1).expand_as(selected_pts_idcs)
    # selected_pts_likelihood = pt_likelihood[SB_helper, ray_helper, selected_pts_idcs]  # SB, N_rays, n_depthsmpls
    selected_pts_likelihood = torch.gather(pt_likelihood, dim=-1, index=selected_pts_idcs)
    zero_liklhd_mask = selected_pts_likelihood == 0.  # pts with 0 likelihood: z_sample=0 for filling up later
    # que_depth
    # z_samples_depth = z_samples[SB_helper, ray_helper, selected_pts_idcs]  # SB, N_rays, N_depthsamples
    z_samples_depth = torch.gather(que_depth.squeeze(0), dim=-1, index=selected_pts_idcs)
    z_samples_depth[zero_liklhd_mask] = 0  # no samples where no depth

    #512, 48

    # gaussian sampling of points
    if n_gaussian > 0:
        # import ipdb;ipdb.set_trace()
        ray_mask = torch.any(opaque_pt_likelihood != 0, dim=-1)  # SB, NR
        z_samples = que_depth.squeeze(0)
        ray_dmean, ray_dstd = weighted_mean_n_std(z_samples[ray_mask],  # B, 1
                                                    opaque_pt_likelihood[ray_mask],
                                                    dim=-1, keepdims=True)
        gauss_samples = torch.zeros(*z_samples.shape[:-1], n_gaussian, device=z_samples.device,
                                    dtype=z_samples.dtype)
        #?
        gauss_samples[ray_mask] = torch.randn_like(gauss_samples[ray_mask]) * ray_dstd + ray_dmean
        # gauss samples: SB, NR, n_gaussian
        z_samples_depth[..., -n_gaussian:] = gauss_samples

    # # visualize pts on target alpha with zero likelihood
    # import matplotlib.pyplot as plt
    # plt.imshow(target_alpha[0][0].cpu())
    # H, W = model.encoder.depths.shape[-2:]
    # plt.scatter(pix_idcs[0][torch.all(zero_liklhd_mask[0], dim=-1)] % W,
    #             pix_idcs[0][torch.all(zero_liklhd_mask[0], dim=-1)] // W,
    #             s=10.)
    # plt.show()

    # ####
    # # Visualize sampling points
    # ####
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    #
    # # draw cameras
    # cam_poses = torch.linalg.inv(model.poses[0]).cpu()
    # s = .1
    # for i, c in enumerate(["red", "green", "blue"]):
    #     ax.quiver(cam_poses[:, 0, -1], cam_poses[:, 1, -1], cam_poses[:, 2, -1],
    #               cam_poses[:, 0, i] * s, cam_poses[:, 1, i] * s, cam_poses[:, 2, i] * s, edgecolor=c)
    # for i in range(len(cam_poses)):
    #     ax.text(cam_poses[i, 0, -1], cam_poses[i, 1, -1], cam_poses[i, 2, -1], str(i))
    #
    # # reprojecting depth maps
    # image_rays = torch.stack(torch.meshgrid(torch.arange(0.5, model.image_shape[1], 1., device=device),
    #                                         torch.arange(0.5, model.image_shape[0], 1., device=device))[::-1],
    #                          dim=-1).reshape(-1, 2)
    # image_rays = image_rays.unsqueeze(0).unsqueeze(0).expand(SB, NV, -1, -1).clone()  # SB, NV, H*W, 2
    # image_rays -= model.c.unsqueeze(-2)
    # image_rays /= model.focal.unsqueeze(-2)
    # image_rays = torch.cat((image_rays, torch.ones_like(image_rays[..., -1:])), dim=-1)  # SB, NV, H*W, 3
    # image_pts = image_rays * model.encoder.depths.view(SB, NV, -1, 1)
    # world_pts = image_pts - model.poses[:, :, :3, -1].unsqueeze(-2)
    # world_pts = torch.matmul(model.poses[:, :, :3, :3].transpose(-2, -1),
    #                          world_pts.transpose(-2, -1)).transpose(-2, -1)
    # world_pts = world_pts.reshape(SB, -1, 3)  # SB, NV*H*W, 3
    #
    # world_pts = world_pts[0][image_pts.reshape(SB, -1, 3)[0, ..., -1] != 0]
    # rand_idcs = torch.randint(len(world_pts), (1000,))
    # ax.scatter(*world_pts[rand_idcs].cpu().unbind(dim=-1), s=5.)
    #
    # # scattering depth_sampled points
    # xyz_sampled_depth = rays[..., None, :3] + z_samples_depth.unsqueeze(-1) * rays[..., None, 3:6]
    # xyz_sampled_depth = xyz_sampled_depth[0][z_samples_depth[0] != 0]
    # ax.scatter(*xyz_sampled_depth.cpu().unbind(dim=-1), s=5.)
    #
    # # printing liklihoods
    # selected_opaque_pt_likelihood = opaque_pt_likelihood[SB_helper, ray_helper, selected_pts_idcs]
    # selected_opaque_pt_likelihood = selected_opaque_pt_likelihood[0][z_samples_depth[0] != 0]
    # for i in range(len(xyz_sampled_depth)):
    #     ax.text(*xyz_sampled_depth[i].cpu().unbind(),
    #             f"{selected_opaque_pt_likelihood[i].cpu().item():.2e}",
    #             fontsize="6")
    #
    # # # scattering naive sampling points
    # # xyz_naive = rays[..., None, :3] + z_samples.unsqueeze(-1) * rays[..., None, 3:6]
    # # xyz_naive = xyz_naive[0].reshape(-1, 3)
    # # rand_idcs = torch.randint(len(xyz_naive), (1000,))
    # # ax.scatter(*xyz_naive[rand_idcs].cpu().unbind(dim=-1), s=5.)
    #
    # ax.set_xlim((-1.5, 1.5))
    # ax.set_ylim((-1.5, 1.5))
    # ax.set_zlim((-1.5, 1.5))
    #
    # # draw reference depth maps
    # nref = model.encoder.depths.shape[1]
    # fig, axes = plt.subplots(ncols=nref, figsize=(nref * 3, 3))
    # for i in range(nref):
    #     dmap = model.encoder.depths[0, i, 0].cpu()
    #     axes[i].imshow(dmap, vmin=dmap[dmap != 0].min())
    #     axes[i].set_title(str(i))
    #
    #     # projecting points on reference images
    #     xyz_sampled_depth_projected = xyz_sampled_depth  # (N, 3)
    #     xyz_sampled_depth_projected_rot = torch.matmul(model.poses[0, i, :3, :3],
    #                                                    xyz_sampled_depth_projected.transpose(-2, -1)).transpose(-2,
    #                                                                                                             -1)
    #     xyz_sampled_depth_projected = xyz_sampled_depth_projected_rot + model.poses[0, i, :3, -1].unsqueeze(
    #         0)  # N, 3
    #     uv = xyz_sampled_depth_projected[..., :2] / xyz_sampled_depth_projected[..., 2:]  # (SB, NV, B, 2)
    #     uv *= model.focal[0, i].unsqueeze(0)
    #     uv += model.c[0, i].unsqueeze(0)
    #     uv = uv.cpu()
    #     axes[i].scatter(uv[:, 0], uv[:, 1], s=5., color="orange")
    #
    # plt.show()
    # import ipdb;ipdb.set_trace()
    new_z_samples_depth = fill_up_uniform_samples(cfg, z_samples_depth).unsqueeze(0)
    return new_z_samples_depth


@torch.no_grad()
def fill_up_uniform_samples(cfg, z_samples):
    """
    Fills up empty slots in samples (indicated by 0) uniformly
    :param z_samples: z values of existing samples (B, n_rays, n_samples). Empty samples have value 0
    # :param rays: (B, n_rays, 8) [origins (3), directions (3), near (1), far (1)]
    :return: filled up z_samples
    """
    # import ipdb;ipdb.set_trace()
    # preparing data to calculate remaining z_samples in parallel
    # ray_shape = rays.shape
    z_samples = z_samples.sort(dim=-1).values  # zeros in the front, important for parallelized filling
    z_samples = z_samples.view(-1, z_samples.shape[-1])  # N_, n_samples
    # rays = rays.view(-1, 8)  # N_, 8
    sample_missing_mask = z_samples == 0  # N_, n_coarsedepth
    missing_iray, missing_isample = torch.where(sample_missing_mask)  # (N_missing,)
    n_missing = sample_missing_mask.int().sum(dim=-1)  # (N_)
    n_missing = n_missing[missing_iray]  # (N_missing,)
    # nears = rays[missing_iray, -2]  # (N_missing,)
    # fars = rays[missing_iray, -1]  # (N_missing,)
    nears = torch.ones_like(n_missing)*cfg["min_depth"]
    fars = torch.ones_like(n_missing)*cfg["max_depth"]

    # calculating remaining z_samples
    step = (fars - nears) / n_missing  # (N_missing,)
    z_missing = nears + missing_isample * step  # (N_missing,)
    z_missing += torch.rand_like(z_missing) * step

    # filling up
    z_samples[missing_iray, missing_isample] = z_missing
    # import ipdb;ipdb.set_trace()
    # z_samples = z_samples.view(*ray_shape[:2], z_samples.shape[-1])
    z_samples = z_samples.sort(dim=-1).values  # sorted z sample values are required in self.composite()
    return z_samples