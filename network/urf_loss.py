import numpy as np
import torch
import math


def compute_urf_loss(args, depth_pr, tvals, weights, target_depth, var_target): #1, 512, 64
    # depth_t = jnp.broadcast_to(batch['depth'], tvals.shape)
    # sigma = (eps / 3.) ** 2

    # eps = torch.sqrt(var) * 3
    # t_from_ndc = 1.0 / (1.0 - tvals)
    # import ipdb;ipdb.set_trace()
    var_t = torch.broadcast_to(var_target.unsqueeze(-1), tvals.shape)
    sigma = torch.sqrt(var_t)
    eps = sigma * 3

    # d_losses = ((depth_pr - target_depth)**2).mean(-1)
    d_losses = (((depth_pr - target_depth)*(target_depth >= args["min_depth"]))**2).mean(-1)

    depth_t = torch.broadcast_to(target_depth.unsqueeze(-1), tvals.shape) #broad to tvals.shape
    depth_p = torch.broadcast_to(depth_pr.unsqueeze(-1), tvals.shape)
    depth_mask = depth_t >= args["min_depth"] #??


    # print("depth_t.shape:", depth_t.shape)
    # print("tvals:", tval.shape)
    # print("eps.shape:", eps.shape)
    # print("weights.shape:", weights.shape)
    mask_near = (tvals > (depth_t - eps)) & (tvals < (depth_t + eps))#.astype(torch.float32)

    mask_near *= depth_mask #.reshape(tvals.shape[0], -1)
    mask_empty = (tvals > (depth_t + eps)) | (tvals < (depth_t - eps)) #.astype(torch.float32)

    mask_empty *= depth_mask #.reshape(tvals.shape[0], -1)
    dist = mask_near * (tvals - depth_t)
    # import ipdb;ipdb.set_trace()
    distr = 1.0 / (sigma * math.sqrt(2 * torch.pi)) * torch.exp(-(dist ** 2 / (2 * sigma ** 2)))
    print("distr.max():", distr.max())
    distr /= distr.max()
    distr *= mask_near

    # n_losses = torch.sum((mask_near * weights - distr) ** 2, dim=-1).mean(-1) #.sum() / jnp.maximum(depth_mask.sum(), 1.0))
    n_losses = torch.sum(mask_near * (weights - distr) ** 2, dim=-1).mean(-1) #.sum() / jnp.maximum(depth_mask.sum(), 1.0))

    e_losses = torch.sum((mask_empty * weights) ** 2, dim=-1).mean(-1) #.sum() / jnp.maximum(depth_mask.sum(), 1.0))

    # import ipdb;ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    # seg_losses.append(((sky_mask.reshape(weights.shape[0], -1) * weights)**2).sum() / jnp.maximum(sky_mask.sum(), 1.0))
    # seg_losses.append((sky_mask * depth).mean())
    # z_from_ndc = depth_mask * (1.0 / (1.0 - depth_mask * depth))
    # print(jnp.max(z_from_ndc))
    # print(jnp.max(depth))
    # inv_depth = depth_mask * (1.0 / jnp.maximum(batch['depth'].squeeze(), 1.0))
    # d_losses = ((depth - batch['depth'].squeeze()) ** 2).mean()

    # return 

    # urf_loss = (n_losses+e_losses).mean(-1)

    # print('urf_loss:', urf_loss)
    # print("d_losses:", d_losses)
    print("empty_loss:", e_losses)
    print("near_loss:", n_losses)
    print("d_loss:", d_losses)
    return args["near_loss_mult"]*n_losses+args["empty_loss_mult"] * e_losses + args["depth_loss_mult"]*d_losses

