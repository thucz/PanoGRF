import torch
import torch.nn.functional as F
from network.ray_utils import get_sphere_ray_directions
import numpy as np
import cv2
debug = True
def points2normal(points):
    # import ipdb;ipdb.set_trace()    
    
    B, _, H, W = points.shape


    point_matrix = F.unfold(points, kernel_size=5, stride=1, padding=4, dilation=2)

    # An = b
    matrix_a = point_matrix.view(B, 3, 25, H, W)  # (B, 3, 25, H, W)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, H, W, 25, 3)
    matrix_a_trans = matrix_a.transpose(3, 4) #a.T
    matrix_b = torch.ones([B, H, W, 25, 1]).cuda()
    # dot(A.T, A)
    point_multi = torch.matmul(matrix_a_trans, matrix_a)
    matrix_deter = torch.det(point_multi.to("cpu")) #det(a.T @ a)

    # make inversible
    inverse_condition = torch.ge(matrix_deter, 1e-5)
    inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
    inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3).cuda()
    # diag matrix to update uninverse
    diag_constant = torch.ones([3], dtype=torch.float32)
    diag_element = torch.diag(diag_constant)
    diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    diag_matrix = diag_element.repeat(B, H, W, 1, 1).cuda()
    # inversible matrix
    inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
    inv_matrix = torch.inverse(inversible_matrix.to("cuda")) #.cuda()

    # import ipdb;ipdb.set_trace()
    ## step.3 compute normal vector use least square
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)

    norm_normalize = F.normalize(generated_norm, p=2, dim=3)
    if debug:
        norm_normalize_np = norm_normalize.squeeze().cpu().numpy()
        # import ipdb;ipdb.set_trace()
        ## step.4 save normal vector
        # np.save(depth_path.replace("depth", "normal"), norm_normalize_np)    
        norm_normalize_draw = (((norm_normalize_np + 1) / 2) * 255).astype(np.uint8)
        cv2.imwrite("normal0.png", norm_normalize_draw[0])
        cv2.imwrite("normal1.png", norm_normalize_draw[1])
        # import ipdb;ipdb.set_trace()
    return norm_normalize

def depth2normal(ref_imgs_info, spt_utils):
    # import ipdb;ipdb.set_trace()
    depthmaps = ref_imgs_info['mvs_depth']#2, 1, h, w
    B, _, H, W = depthmaps.shape

    p_c = get_points_coordinate(depthmaps, ref_imgs_info, spt_utils, 'cpu') #B, 3, H, W
    # normals = points2normal(p_c)#2, H, W, 3, 1
    c2w = ref_imgs_info['c2w'] #B, 3, 4
    # import ipdb;ipdb.set_trace()

    p_c = p_c.view(B, 3, H*W)#B, 3, N
    p_h = torch.cat([p_c, torch.ones(B, 1, H*W).cuda()], dim=1)#B,4, N
    bottom = torch.from_numpy(np.array([[0, 0, 0, 1]])).unsqueeze(0).repeat(B, 1, 1).cuda()
    pose_4x4 = torch.cat([c2w, bottom], dim=1)

    p_w = torch.matmul(pose_4x4, p_h)#B, 4, 4 @ B, 4, N ->B, 4, N
    p_w = p_w[:, :3, :].view(B, 3, H, W)
    normals = points2normal(p_w).squeeze().permute(0, 3, 1, 2)#2, H, W, 3, 1

    # import ipdb;ipdb.set_trace()
    return normals #p_w[:, :3, :].view(B, 3, H, W)

def get_points_coordinate(depth, ref_imgs_info, spt_utils, device="cuda"):
    # import ipdb;ipdb.set_trace()
    # B, height, width, C = depth.size()
    B, c, h, w = depth.size()

    # y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
    #                        torch.arange(0, width, dtype=torch.float32, device=device)])
    # y, x = y.contiguous(), x.contiguous()
    # y, x = y.view(height * width), x.view(height * width)
    # xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]

    # xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    # spt_utils.
    xyz = get_sphere_ray_directions(spt_utils) #h, w, 3
    # import ipdb;ipdb.set_trace()

    xyz = xyz.unsqueeze(0).repeat(B, 1, 1, 1).permute(0, 3, 1, 2)#B,3, h, w
    # import ipdb;ipdb.set_trace()
    # xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H, W]
    depth_xyz = xyz.cuda() * depth #.view(B, 1, -1)  # [B, 3, H, W]
    if debug:
        depth_np = depth.cpu().numpy()
        d_min = depth_np.min()
        d_max = depth_np.max()
        # import ipdb;ipdb.set_trace()

        d_norm = np.uint8((depth_np - d_min)/(d_max-d_min)*255)[0, 0]    
        rgb = np.uint8(ref_imgs_info['imgs'].permute(0, 2, 3, 1)[0].cpu().numpy()*255)
        d_rgb = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        cv2.imwrite("depth0.png", d_rgb)
        cv2.imwrite("rgb0.jpg", rgb)
    return depth_xyz #.view(B, 3, height, width)

# def get_cartesian_coords_camera(spt_utils):
#     # import pdb;pdb.set_trace()
#     pixel_coords = spt_utils.get_xy_coords() #
#     #todo!
#     spherical_coords = spt_utils.equi_2_spherical(pixel_coords) #
#     cartesian_coords = spt_utils.spherical_2_cartesian(spherical_coords)   #cam_pts
    
#     cartesian_coords =  cartesian_coords.view(spt_utils.height, spt_utils.width, 3)
#     cartesian_coords = cartesian_coords / \
#         torch.norm(cartesian_coords, p=2, dim=-1).unsqueeze(-1)
#     return cartesian_coords



