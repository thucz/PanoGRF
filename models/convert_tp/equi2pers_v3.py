import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2

# generate patches in a closed-form
# the transformation and equation is referred from http://blog.nitishmutha.com/equirectangular/360degree/2017/06/12/How-to-project-Equirectangular-image-to-rectilinear-view.html
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def uv2xyz(uv):
    xyz = np.zeros((*uv.shape[:-1], 3), dtype = np.float32)
    xyz[..., 0] = np.multiply(np.cos(uv[..., 1]), np.sin(uv[..., 0]))
    xyz[..., 1] = np.multiply(np.cos(uv[..., 1]), np.cos(uv[..., 0]))
    xyz[..., 2] = np.sin(uv[..., 1])
    return xyz

def equi2pers(erp_img, fov, nrows, patch_size):
    bs, _, erp_h, erp_w = erp_img.shape
    height, width = pair(patch_size)
    fov_h, fov_w = pair(fov)
    FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
    screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)
    
    if nrows==4:    
        num_rows = 4
        num_cols = [3, 6, 6, 3]
        phi_centers = [-67.5, -22.5, 22.5, 67.5]
    if nrows==6:    
        num_rows = 6
        num_cols = [3, 8, 12, 12, 8, 3]
        phi_centers = [-75.2, -45.93, -15.72, 15.72, 45.93, 75.2]
    if nrows==3:
        num_rows = 3
        num_cols = [3, 4, 3]
        phi_centers = [-60, 0, 60]     
    if nrows==5:
        num_rows = 5
        num_cols = [3, 6, 8, 6, 3]
        phi_centers = [-72.2, -36.1, 0, 36.1, 72.2]
            
    phi_interval = 180 // num_rows
    all_combos = []
    erp_mask = []
    for i, n_cols in enumerate(num_cols):
        for j in np.arange(n_cols):
            theta_interval = 360 / n_cols
            theta_center = j * theta_interval + theta_interval / 2

            center = [theta_center, phi_centers[i]]
            all_combos.append(center)
            up = phi_centers[i] + phi_interval / 2
            down = phi_centers[i] - phi_interval / 2
            left = theta_center - theta_interval / 2
            right = theta_center + theta_interval / 2
            up = int((up + 90) / 180 * erp_h)
            down = int((down + 90) / 180 * erp_h)
            left = int(left / 360 * erp_w)
            right = int(right / 360 * erp_w)
            mask = np.zeros((erp_h, erp_w), dtype=int)
            mask[down:up, left:right] = 1
            erp_mask.append(mask)
    all_combos = np.vstack(all_combos) 
    shifts = np.arange(all_combos.shape[0]) * width
    shifts = torch.from_numpy(shifts).float()
    erp_mask = np.stack(erp_mask)
    erp_mask = torch.from_numpy(erp_mask).float()
    num_patch = all_combos.shape[0]

    center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
    center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
    center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

    cp = center_point * 2 - 1
    center_p = cp.clone()
    cp[:, 0] = cp[:, 0] * PI
    cp[:, 1] = cp[:, 1] * PI_2
    cp = cp.unsqueeze(1)
    convertedCoord = screen_points * 2 - 1
    convertedCoord[:, 0] = convertedCoord[:, 0] * PI
    convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
    convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)
    convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

    x = convertedCoord[:, :, 0]
    y = convertedCoord[:, :, 1]

    rou = torch.sqrt(x ** 2 + y ** 2)
    c = torch.atan(rou)
    sin_c = torch.sin(c)
    cos_c = torch.cos(c)
    lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
    lon = cp[:, :, 0] + torch.atan2(x * sin_c, rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
    lat_new = lat / PI_2 
    lon_new = lon / PI 
    lon_new[lon_new > 1] -= 2
    lon_new[lon_new<-1] += 2 

    lon_new = lon_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
    lat_new = lat_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
    grid = torch.stack([lon_new, lat_new], -1)
    grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)

    pers = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
    pers = F.unfold(pers, kernel_size=(height, width), stride=(height, width))
    pers = pers.reshape(bs, -1, height, width, num_patch)
  
    grid_tmp = torch.stack([lon, lat], -1)
    xyz = uv2xyz(grid_tmp)
    xyz = xyz.reshape(num_patch, height, width, 3).transpose(0, 3, 1, 2)
    xyz = torch.from_numpy(xyz).to(pers.device).contiguous()
    
    uv = grid[0, ...].reshape(height, width, num_patch, 2).permute(2, 3, 0, 1)
    uv = uv.contiguous()
    return pers, xyz, uv, center_p

if __name__ == '__main__':
    img = cv2.imread('pano5.png', cv2.IMREAD_COLOR)
    img_new = img.astype(np.float32) 
    img_new = np.transpose(img_new, [2, 0, 1])
    img_new = torch.from_numpy(img_new)
    img_new = img_new.unsqueeze(0)
    pers = equi2pers(img_new, fov=(52, 52), patch_size=(64, 64))
    pers = pers[0].numpy()
    pers = pers.transpose(1, 2, 0).astype(np.uint8)
    cv2.imwrite('pers.png', pers)
    print(pers.shape)