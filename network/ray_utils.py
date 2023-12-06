import torch
from kornia import create_meshgrid

def get_sphere_ray_directions(spt_utils):
    # import pdb;pdb.set_trace()
    pixel_coords = spt_utils.get_xy_coords() #
    #todo!
    # import ipdb;ipdb.set_trace()

    spherical_coords = spt_utils.equi_2_spherical(pixel_coords) #
    cartesian_coords = spt_utils.spherical_2_cartesian(spherical_coords)   #cam_pts
    
    cartesian_coords =  cartesian_coords.view(spt_utils.height, spt_utils.width, 3)
    cartesian_coords = cartesian_coords / \
        torch.norm(cartesian_coords, p=2, dim=-1).unsqueeze(-1)
    return cartesian_coords

def cartesian_2_equi(spt_utils, pts_cam):    
    spherical_coords = spt_utils.cartesian_2_spherical(pts_cam)
    spherical_depth = spherical_coords[:, :, -1]
    pixel_coords = spt_utils.spherical_2_equi(spherical_coords)
    return spherical_depth, pixel_coords

    






def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_sphere_rays(directions, c2w):#directions: cam_pts
    
    # returns ray origin and ray direction in world coordinates
    h, w, _three = directions.shape
    # Ray origin: position of the camera in world coordinates
    # print("c2w.shape::", c2w.shape)
    ray_orig = (c2w[:, 3]).view(1, 1, 3).expand(h, w, 3).view(-1, 3)#??
    # Ray direction in world coordinates
    r_mat = (c2w[:3, :3]).view(1, 1, 3, 3)
    ray_dir = directions.view(h, w, 3, 1)
    
    # import pdb;pdb.set_trace()
    # print("r_mat.dtype:", r_mat.dtype)
    # print("ray_dir.dtype:", ray_dir.dtype)

    ray_dir = torch.matmul(r_mat, ray_dir).view(-1, 3)

    # print("ray_dir.dtype:", ray_dir.dtype)    
    return ray_orig, ray_dir

# def get_rays(directions, c2w):
#     """
#     Get ray origin and normalized directions in world coordinate for all pixels in one image.
#     Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
#                ray-tracing-generating-camera-rays/standard-coordinate-systems

#     Inputs:
#         directions: (H, W, 3) precomputed ray directions in camera coordinate
#         c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

#     Outputs:
#         rays_o: (H*W, 3), the origin of the rays in world coordinate
#         rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
#     """
#     # Rotate ray directions from camera coordinate to the world coordinate
#     rays_d = directions @ c2w[:, :3].T # (H, W, 3)
#     rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
#     # The origin of all rays is the camera origin in world coordinate
#     rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

#     rays_d = rays_d.view(-1, 3)
#     rays_o = rays_o.view(-1, 3)

#     return rays_o, rays_d


# def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
#     """
#     Transform rays from world coordinate to NDC.
#     NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
#     For detailed derivation, please see:
#     http://www.songho.ca/opengl/gl_projectionmatrix.html
#     https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

#     In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
#     See https://github.com/bmild/nerf/issues/18

#     Inputs:
#         H, W, focal: image height, width and focal length
#         near: (N_rays) or float, the depths of the near plane
#         rays_o: (N_rays, 3), the origin of the rays in world coordinate
#         rays_d: (N_rays, 3), the direction of the rays in world coordinate

#     Outputs:
#         rays_o: (N_rays, 3), the origin of the rays in NDC
#         rays_d: (N_rays, 3), the direction of the rays in NDC
#     """
#     # Shift ray origins to near plane
#     t = -(near + rays_o[...,2]) / rays_d[...,2]
#     rays_o = rays_o + t[...,None] * rays_d

#     # Store some intermediate homogeneous results
#     ox_oz = rays_o[...,0] / rays_o[...,2]
#     oy_oz = rays_o[...,1] / rays_o[...,2]
    
#     # Projection
#     o0 = -1./(W/(2.*focal)) * ox_oz
#     o1 = -1./(H/(2.*focal)) * oy_oz
#     o2 = 1. + 2. * near / rays_o[...,2]

#     d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
#     d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
#     d2 = 1 - o2
    
#     rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
#     rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
#     return rays_o, rays_d
