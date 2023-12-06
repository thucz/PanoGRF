from helpers import my_torch_helpers
import torch
import numpy as np
import torch.nn.functional as F
# nghbr_mu, nghbr_sigma, thres
# def calculate_cost_volume_erp(
#                             images,
#                             depths,
#                             trans_norm,
#                             depth_volume=None,
#                             nghbr_mu=None, nghbr_sigma=None, thres=None,
#                             cost_type="abs_diff",
#                             direction="up"):
#     """Calculates a cost volume for ERP images via backwards warping.

#     Panos should be moving forward between images 0 and 1.

#     Args:
#     images: Tensor of shape (B, 2, H, W, C).
#         The target image should be in index 1 along dim 1.
#     depths: Tensor of depths to test.
#     trans_norm: Norm of the translation.
#     cost_type: Type of the cost volume.
#     direction: Direction of cost volume.

#     Returns:
#     Tensor of shape (B, L, H, W, C).
#     """
#     # import pdb;pdb.set_trace()
#     # print("depth_volume.shape:", depth_volume.shape)
#     # import pdb;pdb.set_trace()
#     batch_size, image_ct, height, width, channels = images.shape
#     other_image = images[:, 0]
#     other_image_cf = other_image.permute((0, 3, 1, 2)) #B, C, H, W
#     reference_image_cf = images[:, 1].permute((0, 3, 1, 2))
#     phi = torch.arange(0,
#                     height,
#                     device=images.device,
#                     dtype=images.dtype)
#     phi = (phi + 0.5) * (np.pi / height)
#     theta = torch.arange(0,
#                         width,
#                         device=images.device,
#                         dtype=images.dtype)
#     theta = (theta + 0.5) * (2 * np.pi / width) + np.pi / 2
#     phi, theta = torch.meshgrid(phi, theta)
#     translation = torch.stack(
#         (torch.zeros_like(trans_norm), torch.zeros_like(trans_norm), trans_norm),
#         dim=1)
#     xyz = my_torch_helpers.spherical_to_cartesian(theta, phi, r=1)
#     xyz = xyz[None, :, :, :].expand(batch_size, height, width, 3)#


#     # B = depth_volume.shape[0]
#     nghbr_mu_ = nghbr_mu[:, ...]#B, 1, H, W   #.unsqueeze(0)                                            # D, 1, H, W
#     # nghbr_mu_ = nghbr_mu_.repeat(1, D, 1, 1, 1) #repeat (B, D, 1, 1, 1) ->(B, D, 1, H, W)
#     nghbr_sigma_ = nghbr_sigma[:, ...] #.unsqueeze(1)#.unsqueeze(0)                                      # D, 1, H, W
#     # nghbr_sigma_ = nghbr_sigma_.repeat(1, D, 1, 1, 1)

    
#     cost_volume = []
#     # for i, depth in enumerate(depths): # 64,    
#     #todo
#     D = depth_volume.shape[1]
#     # D = len(depths)

#     for i in range(D):#B, N_s, H, W
        
#         tmp_depth = depth_volume[:, i, :, :]#B, H, W
#         # print("tmp_depth.shape:", tmp_depth.shape)
#         # print("xyz.shape:", xyz.shape)
#         #make sure tmp_depth is the same dimension as xyz.
#         tmp_depth = tmp_depth.unsqueeze(3).repeat(1, 1, 1, 3)


#         # tmp_depth = depths[i]
 
 
#         # import pdb;pdb.set_trace()

#         m_xyz = tmp_depth * xyz - translation[:, None, None, :] # P_cam_src
#         uv = my_torch_helpers.cartesian_to_spherical(m_xyz)
#         # import pdb;pdb.set_trace()
#         u = torch.fmod(uv[..., 0] - np.pi / 2 + 4 * np.pi, 2 * np.pi) / np.pi - 1
#         v = 2 * (uv[..., 1] / np.pi) - 1
#         depth_volume_warped = uv[..., 2]        
#         # print("depth_volume_warped.shape:", depth_volume_warped.shape)#B, H, W
#         # print("other_image_cf.shape:", other_image_cf.shape)
        
#         #cv_image: B, C, H, W(C=Feature dimension,not 3)
#         cv_image = F.grid_sample(
#             other_image_cf,
#             torch.stack((u, v,), dim=-1),
#             mode='bilinear',
#             align_corners=True)


#         # print("nghbr_mu_.shape:", nghbr_mu_.shape)# todo: must make sure the dimension is the same as other_image_cf
#         #nghbr_mu: B, 1, H, W
#         #todo: unify the parameters
#         nghbr_mu_warped = F.grid_sample(nghbr_mu_, torch.stack((u, v,), dim=-1), mode='bilinear', padding_mode='zeros', align_corners=False)#B, 1, H, W
#         nghbr_sigma_warped = F.grid_sample(nghbr_sigma_, torch.stack((u, v,), dim=-1), mode='bilinear', padding_mode='zeros', align_corners=False)#B, 1, H, W

        
#         # if cost_type == 'abs_diff':
#         #     print("cv_image.shape:", cv_image.shape)
#         #     # cv_image = torch.abs(cv_image - reference_image_cf)


#         # elif cost_type != 'none':
#         #     raise ValueError('Unknown cost type')
#         # feat cost:B, H, W
#         # cv_image:B, C, H, W

        
#         # feat_cost =  torch.sum(torch.abs(cv_image - reference_image_cf), axis=1)
        
#         feat_cost = torch.sum((cv_image * reference_image_cf), axis=1)  # (B, H, W)

#         # depth_diff = torch.abs(depth_volume_warped - nghbr_mu_warped[:, 0, :, :])#B, H, W
        
#         # binary_prob = (depth_diff < (nghbr_sigma_warped[:, 0, :, :] * thres)).double()

#         weighted_cost = feat_cost #* binary_prob#B, H, W
        
#         cost_volume.append(weighted_cost)
#     #stack: B, D, H, W
#     cost_volume = torch.stack(cost_volume, dim=1) #.permute((0, 1, 3, 4, 2))

#     # cost_volume = torch.stack(cost_volume, dim=1).permute((0, 1, 3, 4, 2))

#     return cost_volume


def get_cv_per_depth(args, tmp_depth, xyz, batch_size, height, width, tmp_rot_ref, tmp_tran_ref, rot_other, tran_other, other_image_cf, reference_image_cf, cost_type, nghbr_mu_=None, nghbr_sigma_=None, thres=None):
  # B, H, W, 3-> B, H, W, 3, 1
  m_xyz = (tmp_depth * xyz).view(batch_size, -1, 3).unsqueeze(3) #B, H*W, 3, 1

  # rot_ref
  # rot(B, 1, 3, 3) @ ((B, H*W, 3, 1)-(B, 1, 3, 1))->(B, H*W, 3, 1)
  # c2w
  w_xyz = torch.matmul(torch.inverse(tmp_rot_ref), m_xyz - tmp_tran_ref)#world coordinates!
  # xyz_t = torch.matmul(torch.inverse(tmp_rot_ref), tmp_xyz - tmp_tran_ref)#world coordinates!

  # rot_other:B, 3, 3->B, 1, 3, 3 
  # tran_other: B, 3->B, 1, 3
  # w2c
  c_xyz = torch.matmul(rot_other.unsqueeze(1), w_xyz) + tran_other.unsqueeze(1).unsqueeze(3)
  c_xyz = c_xyz.view(batch_size, height, width, 3)

  uv = my_torch_helpers.cartesian_to_spherical(args, c_xyz)#
  # spherical to equi
  if args["dataset_name"] == "m3d":
    u = torch.fmod(uv[..., 0] + np.pi/2 + 2*np.pi, 2 * np.pi) #[0, 2pi]
    v = uv[..., 1]  #[0, pi]
    # u = torch.fmod(uv[..., 0] + np.pi / 2 + 4 * np.pi, 2 * np.pi) / np.pi - 1#revised
    # v = 2 * (uv[..., 1] / np.pi) - 1
    u = u / np.pi - 1 #[-1, 1]
    v = 2 * v / np.pi - 1 #[-1, 1]

  elif args['dataset_name'] == 'replica_test':
    u = torch.fmod(uv[..., 0] + np.pi + 2*np.pi, 2*np.pi) #[0, 2*pi]
    v = -uv[..., 1] + 0.5*np.pi #[-pi/2, pi/2]->[0, np.pi]
    u = u / np.pi - 1 #[-1, 1]
    v = 2 * v / np.pi - 1  #[-1, 1]
  elif args["dataset_name"] == "residential":
    # x_locs = ((1/(2.0*np.pi))*theta + (3/4.0))*(width-1)

    # y_locs = (0.5 - phi/np.pi)*(height-1)
    # import ipdb;ipdb.set_trace()
    u = torch.fmod(uv[..., 0] + 3/4.0*2*np.pi, 2*np.pi) #[0, 2pi]
    v = 0.5*np.pi - uv[..., 1]#[0, np.pi]
    # print("v.max(), v.min():", v.max(), v.min())
    # print("u.max(), u.min():", u.max(), u.min())
    # import ipdb;ipdb.set_trace()
    u = u / np.pi - 1 #[-1, 1]
    v = 2 * v / np.pi - 1  #[-1, 1]
  elif args["dataset_name"] in ["CoffeeArea"]:
    # x_locs = (width-1) * (1 - theta/(2.0*np.pi))
    #         y_locs = phi*(height-1)/np.pi
    u = 2*np.pi - uv[..., 0] #[0, 2pi]
    v = uv[..., 1] #
    # print("v.max(), v.min():", v.max(), v.min())
    # print("u.max(), u.min():", u.max(), u.min())
    # import ipdb;ipdb.set_trace()    
    u = u / np.pi - 1 #[-1, 1]
    v = 2 * v / np.pi - 1  #[-1, 1]    
    # import ipdb;ipdb.set_trace()
  else:
    raise Exception
  assert torch.logical_and(torch.logical_and(u>=-1, u<=1), torch.logical_and(v>=-1, v<=1)).all(),"Wrong UV mapping, UV must be in [-1, 1]!"
  
  cv_image = torch.nn.functional.grid_sample(
    other_image_cf, #reference view -> source views uv
    torch.stack((u, v,), dim=-1),
    mode='bilinear',
    align_corners=True)
  # if args['contain_dnet']:
  #   nghbr_mu_warped = torch.nn.functional.grid_sample(
  #     nghbr_mu_, 
  #     torch.stack((u, v,), dim=-1), 
  #     mode='bilinear', 
  #     align_corners=True)

  #   nghbr_sigma_warped = torch.nn.functional.grid_sample(
  #     nghbr_sigma_, 
  #     torch.stack((u, v,), dim=-1), 
  #     mode='bilinear', 
  #     align_corners=True)
  # if cost_type == 'cost-volume'
  
  if cost_type == 'abs_diff':
    cv_image = torch.abs(cv_image - reference_image_cf)
  elif cost_type=='dot':
    cv_image = cv_image * reference_image_cf
  elif cost_type != 'none':
    raise ValueError('Unknown cost type')
  # import ipdb;ipdb.set_trace()
  # print("tmp_dpeth.shape:", tmp_depth.shape)

  # if args["contain_dnet"]:
  #   depth_volume_warped = uv[..., 2]#uv:B, H, W, 3
  #   # B, H, W
  #   # print("depth_volume_warped.shape:", depth_volume_warped.shape)
  #   depth_diff = torch.abs(depth_volume_warped - nghbr_mu_warped[:, 0, :, :])
  #   binary_prob = (depth_diff < (nghbr_sigma_warped[:, 0, :, :] * thres)).float()#.double()
  #   weighted_cost = cv_image * binary_prob.unsqueeze(1)#B, F, H, W
  #   return weighted_cost
  # else:
  return cv_image

def calculate_cost_volume_erp_multiview(args,
                                images,
                                depths,
                                trans,
                                rots,
                                depth_volume=None,
                                cost_type="abs_diff",
                                ref_gmms=None,
                                nghbr_gmms = None,
                                thres=None,
                                direction="up",
                                curr_idx=0):
    """
    Calculates a cost volume for ERP images via backwards warping.

    Panos should be moving forward between images 0 and 1.

    Args:
      images: Tensor of shape (B, seq_len, H, W, C).
        The target image should be in index 1 along dim 1.
      depths: Tensor of depths to test.
      trans_norm: Norm of the translation.
      cost_type: Type of the cost volume.
      direction: Direction of cost volume.
      rots: B, L, 3, 3
      trans:B, L, 3   
    Returns:
      Tensor of shape (B, L, H, W, C).
    """
    batch_size, image_ct, height, width, channels = images.shape
    # import ipdb;ipdb.set_trace()
    # rots = rots.view(batch_size, image_ct, 3, 3)
    # trans = trans.view(batch_size, image_ct, 3)
    other_image = images.clone() # include reference view
    other_image_cf = other_image.permute((0, 1, 4, 2, 3))      # source views
    reference_image_cf = images[:, curr_idx].permute((0, 3, 1, 2)) # reference view    
    if args['contain_dnet']:
      if args["mono_uncertainty"]:
        nghbr_mu, nghbr_sigma = torch.split(nghbr_gmms, 1, dim=1)  # BxV, 1, H, W
      else:
        nghbr_mu = nghbr_gmms #torch.split(nghbr_gmms, 1, dim=1)  # BxV, 1, H, W


    phi = torch.arange(0,
                       height,
                       device=images.device,
                       dtype=images.dtype)
    theta = torch.arange(0,
                         width,
                         device=images.device,
                         dtype=images.dtype)
    # equi2spherical
    if args["dataset_name"] == "m3d":
      phi = (phi + 0.5) * (np.pi / height)    
      theta = (theta + 0.5) * (2 * np.pi / width) - np.pi / 2#revised
    elif args["dataset_name"] == "replica_test":
      theta = (2*np.pi / width) * (theta+0.5) - np.pi
      # phi = (np.pi/height)*(phi+0.5)
      phi = -(phi+0.5)*np.pi/height+np.pi*0.5
    elif args["dataset_name"] == "residential":
      theta = np.pi*(2*theta/(width-1) - 1.5)
      phi = np.pi*(0.5-phi/(height-1))
    elif args["dataset_name"] in ["CoffeeArea"]:
      theta = (-2*np.pi / (width-1)) * theta + 2*np.pi
      phi = (np.pi/(height-1))*(phi)
    else:
      raise Exception
            


    phi, theta = torch.meshgrid(phi, theta)
    xyz = my_torch_helpers.spherical_to_cartesian(args, theta, phi, r=1) #spherical to camera cartesian    
    xyz = xyz[None, :, :, :].expand(batch_size, height, width, 3)
    
    rot_other = rots.clone()
    tran_other = trans.clone()
    rot_ref = rots[:, curr_idx]
    tran_ref = trans[:, curr_idx]
    tmp_tran_ref = tran_ref.unsqueeze(1).unsqueeze(3)#B, 3->B, 1, 3, 1
    tmp_rot_ref = rot_ref.unsqueeze(1)#B, 3, 3->B, 1, 3, 3          

  
    # reference -> source
    # c2w->w2c
    # import ipdb;ipdb.set_trace()

    # rot@w+trans=C
    # rot^(-1)@(c-trans)->w
    
    if args["contain_dnet"]:
      D = depth_volume.shape[1]
      seq_len = images.shape[1]
      mv_cost_volume = 0
      for view_id in range(seq_len-1):#exclude the last (center viewpoint)
        if view_id == curr_idx:# do not compare with itself
          continue
        cost_volume = []    
        for depth_id in range(D):
          tmp_depth = depth_volume[:, depth_id, :, :]#B, H, W
          #make sure tmp_depth is the same dimension as xyz.
          tmp_depth = tmp_depth.unsqueeze(3).repeat(1, 1, 1, 3)#B, H, W, 3
          #depth_volume
          # m_xyz = tmp_depth * xyz - translation[:, None, None, :] #c2w
          # import ipdb;ipdb.set_trace()
          # nghbr_mu_ = nghbr_mu#.repeat(D, 1, 1, 1)
          # if args["mono_uncertainty"]:
          #   nghbr_sigma_ = nghbr_sigma#.repeat(D, 1, 1, 1)        
          # cv_image = get_cv_per_depth(args, tmp_depth, xyz, batch_size, height, width, tmp_rot_ref, tmp_tran_ref, rot_other, tran_other, other_image_cf, reference_image_cf, cost_type, nghbr_mu_, nghbr_sigma_, thres=thres)
          cv_image = get_cv_per_depth(args, tmp_depth, xyz, batch_size, height, width, tmp_rot_ref, tmp_tran_ref, rot_other[:, view_id], tran_other[:, view_id], other_image_cf[:, view_id], reference_image_cf, cost_type)
          cost_volume.append(cv_image)
        cost_volume = torch.stack(cost_volume, dim=1).permute((0, 1, 3, 4, 2))#B, D, H, W, C
        assert seq_len>2
        mv_cost_volume += cost_volume/(seq_len-2)#seq_len>2
    else:
      # D = depth_volume.shape[1]
      seq_len = images.shape[1]
      mv_cost_volume = 0      
      for view_id in range(seq_len-1):#exclude the last (center viewpoint)
        if view_id == curr_idx:# do not compare with itself
          continue      
        cost_volume = []    
        for i, tmp_depth in enumerate(depths):
          # reference view -> source views uv
          cv_image = get_cv_per_depth(args, tmp_depth, xyz, batch_size, height, width, tmp_rot_ref, tmp_tran_ref, rot_other[:, view_id], tran_other[:, view_id], other_image_cf[:, view_id], reference_image_cf, cost_type)    
          cost_volume.append(cv_image)
        cost_volume = torch.stack(cost_volume, dim=1).permute((0, 1, 3, 4, 2))#B, D, H, W, C
        assert seq_len>2
        mv_cost_volume += cost_volume/(seq_len-2)#seq_len>2
    return mv_cost_volume

