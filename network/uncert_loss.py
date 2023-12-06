import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def is_not_in_expected_distribution(depth_mean, depth_var, depth_measurement_mean, depth_measurement_var): #depth_measurement_std):
    # delta_greater_than_expected = ((depth_mean - depth_measurement_mean).abs() - depth_measurement_std) > 0.
    delta_greater_than_expected = ((depth_mean - depth_measurement_mean).abs() - torch.sqrt(depth_measurement_var) ) > 0.
    # var_greater_than_expected = depth_measurement_std.pow(2) < depth_var
    var_greater_than_expected = depth_measurement_var < depth_var
    return torch.logical_or(delta_greater_than_expected, var_greater_than_expected)


def compute_nll_loss(args, pred_mean, pred_var_in, target_mean, target_var, clip_sigma=0.0, weights_gt=None, bk_valid_mask=None): #, target_valid_depth):
    # target_mean = target_depth[..., 0]#[target_valid_depth]
    # # target_std = target_depth[..., 1][target_valid_depth]
    # target_var = target_depth[..., 1]#[target_valid_depth]
    # import ipdb;ipdb.set_trace()
    pred_var = torch.square(torch.clamp(torch.sqrt(pred_var_in), min=clip_sigma))
    if "apply_all" in args and args["apply_all"]:
        apply_depth_loss = torch.ones_like(pred_mean) #is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_var) #target_std
    else:
        apply_depth_loss = is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_var) #target_std
    # debug=True
    # if debug:
    #     bk_valid_mask = None
        # bk_valid_mask = torch.ones_like(bk_valid_mask)
    # if target_mean 
    # apply_depth_loss, 
    # import ipdb;ipdb.set_trace()
    # only for valid depth
    apply_depth_loss = torch.logical_and(torch.logical_and(apply_depth_loss, target_mean > args["min_depth"]), target_mean < args["max_depth"])

    # pred_mean = pred_mean[apply_depth_loss]
    # pred_var = pred_var[apply_depth_loss]
    # target_mean = target_mean[apply_depth_loss]
    # target_std = target_std[apply_depth_loss]
    # target_var = target_var[apply_depth_loss]    
    f = nn.GaussianNLLLoss(eps=0.001, reduction='none')
    # apply_depth_loss
    # return float(pred_mean.shape[0]) / float(target_valid_depth.shape[0]) * f(pred_mean, target_mean, pred_var)
    # import ipdb;ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    if bk_valid_mask is not None:
        # import ipdb;ipdb.set_trace()
        if weights_gt is not None:
            return  torch.sum(f(pred_mean, target_mean, pred_var) * apply_depth_loss * weights_gt * bk_valid_mask, dim=1)/(torch.sum(weights_gt*bk_valid_mask*apply_depth_loss, dim=1) + 1e-7)
        else:
            return  torch.sum(f(pred_mean, target_mean, pred_var) * apply_depth_loss * bk_valid_mask, dim=1)/ (torch.sum(apply_depth_loss * bk_valid_mask, dim=1) + 1e-7)
    else:
        if weights_gt is not None:
            return  torch.sum(f(pred_mean, target_mean, pred_var) * apply_depth_loss * weights_gt, dim=1)/(torch.sum(apply_depth_loss * weights_gt, dim=1)+1e-7)
        else:
            return  torch.sum(f(pred_mean, target_mean, pred_var) * apply_depth_loss, dim=1 )/(torch.sum(apply_depth_loss, dim=1) + 1e-7)


def compute_perpoint_loss(args, pred_dvals, weights, target_depth, weights_gt=None):
    depth_mask = target_depth.unsqueeze(-1)>=args["min_depth"]
    if "acc" in args and args["acc"]:
        acc = 1-torch.sum(weights, -1)
        acc_loss = acc.mean(dim=1)

    perpoint_loss= torch.sum(weights*depth_mask*((pred_dvals - target_depth.unsqueeze(-1))**2), dim=-1)
    if weights_gt is not None:

        perpoint_loss = torch.sum(perpoint_loss * weights_gt, dim=1)/torch.sum(weights_gt, dim=1)
        if "acc" in args and args["acc"]:
            perpoint_loss = perpoint_loss +acc_loss
        return perpoint_loss
    else:
        if "acc" in args and args["acc"]:
            return perpoint_loss.mean(dim=1) + acc_loss
        else:
            return perpoint_loss.mean(dim=1) 

    # return loss 
#URF
def compute_urf_loss(pred_dvals, weights, target_depth):

    pass






    