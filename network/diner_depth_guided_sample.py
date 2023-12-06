import math
import torch
from network.sample_utils import sample_3sigma
#
# def get_prob_per_view(cfg):
    
#     # mu, sigma known:
#     delta = (far - near) / N_candidates
#     #per view likelihood
#     prob_perview =  delta * 1.0 / (torch.sqrt(2*math.pi) * sigma) * torch.exp(-0.5*((t_vals - mu)/sigma).pow(2))

# 1. project_to_per_view()
# 2. get_prob_per_view()

#. 2. multi-view likelihood:
# p_perview = torch.max(p_perview, dim=1)
# shortlist

# sample N_samples
# 4. occlusion-aware likelihood:

# first sort and then cumsum
# p_perview, index = torch.sort(p_perview) # 按照depth排序

# occ_p = 1 - p_perview # note:这里的p_perview可能是对1000个点
# occ_cum_p = torch.cumprod(occ_p, dim=-1) # 假设最后一维
# p_oa = occ_cum_p * p_perview / (1-p_perview) # p_xi * cumprod(1-p_xj), where xj < xi

def select_depth(cfg, prj_depth_info_dict, que_depth, que_dir, include_norm=False, var=True):
    # import ipdb;ipdb.set_trace()
    que_depth = que_depth.cpu()
    mu = prj_depth_info_dict['ref_mvs_depths'].squeeze(-1).cpu()
    uncert = prj_depth_info_dict['ref_mvs_uncert'].squeeze(-1).cpu()
    prj_depth = prj_depth_info_dict['depth'].squeeze(-1).cpu() #?
    if include_norm:
        prj_norm = prj_depth_info_dict['ref_mvs_normal'].squeeze(-1).cpu() #?

    # print('que_dir.shape:', que_dir.shape)#1, 512, 1000, 3
    # print("prj_norm.shape:", prj_norm.shape)#2, 1, 512, 1000, 3
    # import ipdb;ipdb.set_trace()
    rectified_que_dir = que_dir #?

    # print('mu.shape:', mu.shape)
    if var:
        sigma = torch.sqrt(uncert)
    else:
        sigma = uncert

    if "diner_fix_uncert" in cfg and cfg['diner_fix_uncert']:
        sigma = cfg["diner_fix_uncert"]
        
    # mu, sigma known:
    delta = (cfg["max_depth"] - cfg["min_depth"]) / cfg["N_diner"] #N_diner: 1000

    # 1.per-view likelihood 
    # import ipdb;ipdb.set_trace()
    prob_perview = delta * 1.0 / (math.sqrt(2*math.pi) * sigma) * torch.exp(-0.5*((prj_depth - mu)/sigma).pow(2))

    # import ipdb;ipdb.set_trace()
    #1, 512, 1000, 3
    # backface_culling(rectified_que_dir, prj_norm) #2, 1, 512, 1000, 3
    if include_norm:
        B = prob_perview.shape[0]
        rectified_que_dir = rectified_que_dir.unsqueeze(0).repeat(B, 1, 1, 1, 1) 
        cos = torch.sum(rectified_que_dir.cpu() * prj_norm, dim=-1) #2, 1, 512, 1000, 3
        eps = 1e-3
        zero_mat = torch.zeros_like(prob_perview)
        # new_prob_perview = torch.where(cos > eps, zero_mat, prob_perview)# < 90
        new_prob_perview = torch.where(cos > eps, zero_mat, prob_perview)# < 90
        
    else:
        new_prob_perview = prob_perview
    # rfn, qn, rn, dn, 1
    # 2. todo?:backface culling(calculate_normals):    
    #. 2. multi-view likelihood:

    p_perview, _ = torch.max(new_prob_perview, dim=0) #dim ?
    #1, 512, 1000, 1
    # sc    
    # import ipdb;ipdb.set_trace()
    # shortlist
    sort_prob_perview, index = torch.sort(p_perview, dim=2, descending=True) #dim?

    new_que_depth = torch.gather(que_depth, dim=2, index=index) #index dim?
    
    # sort_prob_perview
    new_que_depth = new_que_depth[..., :cfg["N_sample"]] #init depth samples
    
    
    occ_p = 1 - p_perview # note:这里的p_perview可能是对1000个点
    occ_cum_p = torch.cumprod(occ_p, dim=-1) # 假设最后一维
    p_oa = occ_cum_p * p_perview / (1-p_perview) # p_xi * cumprod(1-p_xj), where xj < xi

    # index = torch.where(p_oa, )
    # p_oa
    # eps = 1e-2
    # pos = torch.where( p_oa > 1 - eps )
    # index = torch.argwhere(p_oa > 1-eps)
    # torch.argmax(p_oa, dim=-1)
    # hypo = que_depth[None, None, None, :]
    if torch.any(torch.sum(p_oa, -1)==0):
        import ipdb;ipdb.set_trace()


    terminate_expect = torch.sum(p_oa * que_depth, dim=-1) / torch.sum(p_oa, dim=-1) 


    #sample_around terminate_expectation: N_guass
    target_sigma = cfg["target_sigma"] #todo?
    low_3sigma = terminate_expect - 3*target_sigma
    high_3sigma = terminate_expect + 3*target_sigma
    det = True
    # import ipdb;ipdb.set_trace()
    guass_hypo = sample_3sigma(low_3sigma.squeeze(0).cuda(), high_3sigma.squeeze(0).cuda(), cfg["N_guass"], det, cfg["min_depth"], cfg["max_depth"]).unsqueeze(0)

    # pass
    final_depth = torch.cat([new_que_depth.cuda(), guass_hypo], dim=-1)
    final_depth, _ = torch.sort(final_depth, dim=-1, descending=False)
    # if torch.isnan(final_depth).any():
    #     print("contain!")
    #     import ipdb;ipdb.set_trace()
    # else:
    #     print("not contain!")
    #     import ipdb;ipdb.set_trace()

    return final_depth


