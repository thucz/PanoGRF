import os
import cv2 as cv
import sys
import copy
import math
import torch
import numpy as np
import imageio
import argparse
from string import Template
import torch.nn.functional as F
import h5py
# sys.path.append('../src/models')
# from spt_utils import Utils


class Configs:
    def __init__(self):
        self.dataset = 'd3dkit'
        self.batch_size = 1
        self.height = 512
        self.width = 1024


parser = argparse.ArgumentParser()

# #!/bin/bash
# image_path=./sample_data/CoffeeAreaR1
# # rm -r $image_path/perspective
# python3 equi2perspective.py \
#         -i=$image_path/images \
#         -o=$image_path/perspective/images \
#         --pose_out_path=$image_path/perspective/poses_to_sphere \
#         --fov 90 \
#         --theta 180 \
#         --phi 80

# cubemap_rotations = [
# Rotation.from_euler('x', 90, degrees=True),  # Top
# Rotation.from_euler('y', 0, degrees=True), #front
# Rotation.from_euler('y', -90, degrees=True), # left
# Rotation.from_euler('y', -180, degrees=True), #back
# Rotation.from_euler('y', -270, degrees=True), #right
# Rotation.from_euler('x', -90, degrees=True)  # Bottom
# ]

#phi, theta:
#1.(90, 0)
#2.(0, 0)
#3.(0, -90)
#4.(0, -180)
#5.(0, -270)
#6.(-90, 0)
# subid_dict = {
#     (90, 0):   0,
#     (0, 0):    1,
#     (0, -90):  2,
#     (0, -180): 3,
#     (0, -270): 4,
#     (-90, 0):  5,
# }

parser.add_argument('-p', '--phi', type=float, default=90)
parser.add_argument('-t', '--theta', type=float, default=0)
parser.add_argument('-f', '--fov', type=float, default=90)
parser.add_argument('-d','--dataset', type=str, default='residential')#d3dkit')
parser.add_argument('-s','--scene_dir', type=str, default='')#d3dkit')

parser.add_argument('-i', '--input_path', type=str, default='/mnt/data1/chenzheng/tmp/residential/ricoh_mini')
parser.add_argument('-o', '--out_path', type=str, default='/mnt/data1/chenzheng/tmp/residential/ricoh_mini/perspective/images')
parser.add_argument('--pose_out_path', type=str, default='/mnt/data1/chenzheng/tmp/residential/ricoh_mini/perspective/poses_to_sphere')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--scene_number', type=int, default=1)

args = parser.parse_args()
args_dict = dict(args.__dict__)
def merge_r_and_t(r, t):
    b = r.shape[0]
    # import ipdb;ipdb.set_trace()
    c2w = torch.cat([torch.cat([r, t], dim=2), torch.FloatTensor(
        [0, 0, 0, 1]).view(1, 1, 4).expand(b, 1, 4)], dim=1)
    return c2w

class PerspectiveCutout:
    def __init__(self, opts):
        self.opts = opts
        self.configs = Configs()
        self.device = 'cuda:0' if opts.device.startswith('cuda') else 'cpu'
        self.dataset = self.opts.dataset
        self.adjust_to_training = True

    def get_K(self, fov, h, w):
        if self.adjust_to_training:
            ph = h//2
            pw = h//2
            f = 0.5 * pw * 1 / np.tan(0.5 * fov / 180.0 * np.pi) #
            fx = fy = f    
        else:
            fx, fy = w / (2*math.pi), h / math.pi
            ph = int(2*fy*math.tan(math.radians(fov/2)))#perspective height
            #fy = 0.5*ph
            pw = int(2*fx*math.tan(math.radians(fov/2)))
        cy = (ph-1) * 0.5
        cx = (pw-1) * 0.5
            
        K = torch.from_numpy(np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,   1],
        ], np.float32))
        return K, fx, fy, ph, pw

    def get_perspec(self, s2w, rot_mats): #c2w 
        #r_mats: n, 3, 3
        #t_vecs: n, 3, 1
        #rot_mats: n, 3, 3 
        # import ipdb;ipdb.set_trace()
        # r_mats = torch.stack(r_mats, dim=0)
        # t_vecs = torch.stack(t_vecs, dim=0)
        rot_mats=torch.stack(rot_mats, dim=0)
        r_s2c = rot_mats
        # c2w = merge_r_and_t(r, t)
        b = rot_mats.shape[0]
        t_s2c = torch.zeros(b, 3, 1).float()
        s2c = merge_r_and_t(r_s2c, t_s2c)
        # s2w = torch.bmm(s2c, c2w)
        # s2w = merge_r_and_t(r_mats, t_vecs)
        c2w = torch.bmm(s2c.transpose(-2, -1), s2w)
        return c2w

    def __call__(self):


        # theta = self.opts.theta
        # phi = self.opts.phi
        subid_dict = {
            ( 0, 90):   0,
            ( 0, 0):    1,
            (-90, 0):  2,
            (-180, 0): 3,
            (-270, 0): 4,
            (0, -90):  5,
        }
        # r_mats, t_vecs = self.read_poses()
        # s2w = merge_r_and_t(r_mats, t_vecs)    
        images, poses = self._read_data(self.opts.input_path)
        s2w = poses
        #intrinsics for perspective
        im = images[0].unsqueeze(0)
        b, _c, h, w = im.shape
        # configs = copy.deepcopy(self.configs)
        print("h, w:", h, w)
        # configs.height, configs.width = h, w
        # configs.batch_size, configs.dataset = b, self.opts.dataset
        fov = self.opts.fov

        K, fx, fy, ph, pw = self.get_K(fov, h, w)
        self.fx = fx
        self.fy = fy
        self.ph = ph
        self.pw = pw


        perspec_c2w_all = []

        perspec_imgs_all = []

        for key, value in subid_dict.items():
            print("key:", key)
            phi, theta = key[0], key[1]
            sub_id = value
            cut_outs_rot_mats = [self.cut(im, theta, phi) for im in images]
            cut_outs = [f[0] for f in cut_outs_rot_mats]
            rot_mats = [(f[1]).to('cpu') for f in cut_outs_rot_mats]
            perspec_imgs_all.append(torch.stack(cut_outs, dim=0))
            perspec_c2w = self.get_perspec(s2w, rot_mats)
            perspec_c2w_all.append(perspec_c2w)
            self.save_images(cut_outs, self.opts.out_path, sub_id)
            # self.save_poses(rot_mats, self.opts.out_path)
        # torch.save(+'r_mats.t7')
        perspec_c2w_all = torch.stack(perspec_c2w_all, dim=1)
        # import ipdb;ipdb.set_trace()
        perspec_imgs_all = torch.stack(perspec_imgs_all, dim=1)#BGR, 0~1
        # rgb->c2w:
        # 0->5
        # 1->1
        # 2->4
        # 3->3
        # 4->2
        # 5->0

        data_dict={
            "c2w": s2w, #sphere:c2w
            "rgbs": images,
            "cube_c2w": perspec_c2w_all,
            "cube_rgbs": perspec_imgs_all,
            "K": K
        }

        torch.save(data_dict, os.path.join(self.opts.scene_dir, "all.t7"))

    def get_inverse_rotation_for_center(self, phi, theta, device='cuda:0'):
        # rz = self.get_rotation_z(90 - theta)
        ry = self.get_rotation_y(theta)
        rx = self.get_rotation_x(phi)
        return (ry.permute(1, 0) @ rx.permute(1, 0)).to(device)
        # return (rx @ rz).to(device)


    def cut(self, im, theta, phi):
        im = im.unsqueeze(0)
        b, _c, h, w = im.shape
        configs = copy.deepcopy(self.configs)
        configs.height, configs.width = h, w
        configs.batch_size, configs.dataset = b, self.opts.dataset
        # f_x, f_y = w / (2*math.pi), h / math.pi
        # rays = self.get_rays_at_z(self.opts.fov, f_x, f_y).to(im.device)
        rays = self.get_rays_at_z(self.opts.fov).to(im.device)

        out_h, out_w = rays.shape[1], rays.shape[2]
        rays = rays.unsqueeze(-1).view(-1, 3, 1)
        rotation = self.get_inverse_rotation_for_center(phi, theta, im.device)
        rays_spherical_coords = torch.bmm(
            rotation.view(-1, 3, 3).expand(rays.shape[0], 3, 3), rays.view(-1, 3, 1))
        rays_spherical_coords = rays_spherical_coords.view(b, out_h, out_w, 3)
        rays_spherical = self.cartesian_2_spherical(rays_spherical_coords)
        rays_equi = self.spherical_2_equi(rays_spherical, h, w)
        rays_equi[..., 0] = rays_equi[..., 0] - (w-1)/2.0
        rays_equi[..., 0] = rays_equi[..., 0] / ((w-1)/2.0)
        rays_equi[..., 1] = rays_equi[..., 1] - (h-1)/2.0
        rays_equi[..., 1] = rays_equi[..., 1] / ((h-1)/2.0)
        out = F.grid_sample(im.view(1, 3, h, w), grid=rays_equi, mode='bilinear', align_corners=True)
        return out, rotation

    def spherical_2_equi(self, spherical_coords, height, width):
        last_coord_one = False
        if spherical_coords.shape[-1] == 1:
            spherical_coords = spherical_coords.squeeze(-1)
            last_coord_one = True
        spherical_coords = torch.split(
            spherical_coords, split_size_or_sections=1, dim=-1)
        theta, phi = spherical_coords[0], spherical_coords[1]
        if 'replica' in self.dataset:
            x_locs = ((width-1)/(2.0*math.pi)) * (theta + math.pi)
            y_locs = (height-1)/math.pi * phi
        elif self.dataset == 'residential':
            x_locs = ((1/(2.0*math.pi))*theta + (3/4.0))*(width-1)
            y_locs = (0.5 - phi/math.pi)*(height-1)
        else:
            x_locs = (width-1) * (1 - theta/(2.0*math.pi))
            y_locs = phi*(height-1)/math.pi
        # x_locs = (width-1) * (1 - theta/(2.0*math.pi))
        # y_locs = phi*(height-1)/math.pi
        
        xy_locs = torch.cat([x_locs, y_locs], dim=-1)
        if last_coord_one:
            xy_locs = xy_locs.unsqueeze(-1)

        return xy_locs


    def normalize_3d_vectors(self, input_points, p=2, eps=1e-12):
        '''normalises input 3d points along the last dimension
        :param input_points: 3D points of shape [B, ..., 3]
        :param p: norm power
        :param eps: epsilone to avoid division by 0
        '''
        input_shape = input_points.shape
        last_coord_one = False
        if input_shape[-1] == 1:
            last_coord_one = True
            input_points = input_points.squeeze(-1)
        p_norm = torch.norm(input_points, p=p, dim=-1,
                            keepdim=True).clamp(min=eps)
        normalized_points = input_points / p_norm
        if last_coord_one:
            normalized_points = normalized_points.unsqueeze(-1)
        return normalized_points

    def cartesian_2_spherical(self, input_points, normalized=False):
        last_coord_one = False
        if input_points.shape[-1] == 1:
            input_points = input_points.squeeze(-1)
            last_coord_one = True
        if not normalized:
            input_points = self.normalize_3d_vectors(input_points)
        x_c, y_c, z_c = torch.split(
            input_points, split_size_or_sections=1, dim=-1)
        r = torch.sqrt(x_c**2 + y_c**2 + z_c**2)
        if 'replica' in self.dataset:
            theta = torch.atan2(y_c, x_c)
            phi = torch.acos(z_c/r)
            mask1 = theta.gt(math.pi)
            theta[mask1] = theta[mask1] - 2*math.pi
            mask2 = theta.lt(-1*math.pi)
            theta[mask2] = theta[mask2] + 2*math.pi
        elif self.dataset == 'residential':
            theta = -torch.atan2(-z_c, x_c)
            phi = torch.asin(y_c/r)
            mask = torch.logical_and(
                theta.gt(math.pi*0.5), theta.le(2*math.pi))
            theta[mask] = theta[mask] - 2*math.pi
        else:
            theta = torch.atan2(y_c, x_c)
            phi = torch.acos(z_c/r)
            mask1 = theta.lt(0)
            theta[mask1] = theta[mask1] + 2*math.pi
        # theta = torch.atan2(y_c, x_c)
        # phi = torch.acos(z_c/r)
        # mask1 = theta.lt(0)
        # theta[mask1] = theta[mask1] + 2*math.pi
        # mask2 = theta.lt(0)
        # theta[mask2] = theta[mask2] + 2*math.pi
        spherical_coords = torch.cat(
            [theta, phi, torch.ones_like(theta)], dim=-1)
        # spherical to equi
        return spherical_coords

    def get_rotation_x(self, angle, device='cuda:0'):
        print(angle)
        angle = math.radians(angle)
        sin, cos = math.sin(angle), math.cos(angle)
        r_mat = torch.eye(3).to(device)
        r_mat[1, 1] = cos
        r_mat[1, 2] = -sin
        r_mat[2, 1] = sin
        r_mat[2, 2] = cos
        return r_mat
    def get_rotation_y(self, angle, device='cuda:0'): #counterclockwise rotation around y-axis
        print('***', angle)
        angle = math.radians(angle)
        sin, cos = math.sin(angle), math.cos(angle)
        r_mat = torch.eye(3).to(device)
        r_mat[0, 0] = cos
        r_mat[0, 2] = sin
        r_mat[2, 0] = -sin
        r_mat[2, 2] = cos
        return r_mat

    def get_rotation_z(self, angle, device='cuda:0'):
        print('***', angle)
        angle = math.radians(angle)
        sin, cos = math.sin(angle), math.cos(angle)
        r_mat = torch.eye(3).to(device)
        r_mat[0, 0] = cos
        r_mat[0, 1] = -sin
        r_mat[1, 0] = sin
        r_mat[1, 1] = cos
        return r_mat

    def rotate_z(self, rays, angle):
        angle = math.radians(angle)
        sin, cos = math.sin(angle), math.cos(angle)
        r_mat = torch.eye(3).to(rays.device)
        r_mat[0, 0] = cos
        r_mat[0, 1] = -sin
        r_mat[1, 0] = sin
        r_mat[1, 1] = cos
        #
        rotated_rays = torch.matmul(r_mat.view(1, 1, 1, 3, 3), rays)
        return rotated_rays

    def rotate_y(self, rays, angle):
        angle = math.radians(angle)
        sin, cos = math.sin(angle), math.cos(angle)
        r_mat = torch.eye(3).to(rays.device)
        r_mat[0, 0] = cos
        r_mat[0, 2] = sin
        r_mat[2, 0] = -1*sin
        r_mat[2, 2] = cos
        rotated_rays = torch.matmul(r_mat.view(1, 1, 1, 3, 3), rays)
        return rotated_rays


    def get_rays_at_z(self, fov):
        b = 1
        # h = int(2*fy*math.tan(math.radians(fov/2)))
        # w = int(2*fx*math.tan(math.radians(fov/2)))
        h = self.ph
        w = self.pw
        fx = self.fx
        fy = self.fy
        y_locs = torch.linspace(0, h-1, h).to(self.device)
        x_locs = torch.linspace(0, w-1, w).to(self.device)
        x_locs = x_locs.view(1, 1, w, 1).expand(b, h, w, 1)
        y_locs = y_locs.view(1, h, 1, 1).expand(b, h, w, 1)
        # compute rays
        x_locs = (x_locs - (w)*0.5) / fx
        y_locs = (y_locs - (h)*0.5) / fy
        ones = torch.ones(b, h, w, 1).to(self.device)
        # swap x and y to fit the coordinates of the spherical datasets
        # rays = torch.cat([y_locs, -1*x_locs, ones], dim=3)
        rays = torch.cat([x_locs, y_locs, ones], dim=3)
        return rays

    def save_images(self, images, out_path, sub_id):
        images = list(images) if not isinstance(images, list) else images
        os.makedirs(self.opts.out_path, exist_ok=True)
        # pre_fix = Template('./out_$num.png')
        for i in range(len(images)):
            fname = "./out_"+str(i).zfill(5)+"_"+str(sub_id)+".png"
            # fname = pre_fix.substitute(num=str(i).zfill(5))
            img = (images[i]).squeeze().cpu()
            img = np.asarray(img.permute(1, 2, 0).numpy()*255, dtype=np.uint8)
            # import ipdb;ipdb.set_trace()
            cv.imwrite(os.path.join(self.opts.out_path, fname), img)


    def _read_data(self, data_path):
        file_name = os.path.join(data_path, f'{self.opts.scene_number}.h5')    
        w, h = self.configs.width, self.configs.height#self.img_wh
        ref_idx = 4
        with h5py.File(file_name, 'r') as f:        
            color = torch.from_numpy(f['color'][:]).float().contiguous()
            color = color/255
            color = color.permute(0, 3, 1, 2)
            color = F.interpolate(color, size=(h, w), align_corners=False, mode='bilinear')
            # color = color.permute(0, 2, 3, 1)
            pose = torch.from_numpy(f['pose'][:]).float().contiguous()
            pose = self._adjust_world(pose, ref_idx)[..., :4, :4]
        return color, pose

    def _adjust_world(self, pose, ref_idx):
        mid_idx = ref_idx
        ref_pose = (pose[mid_idx]).view(1, 4, 4).expand_as(pose)
        pose = torch.bmm(torch.inverse(ref_pose), pose)
        return pose



if __name__ == '__main__':
    perpective = PerspectiveCutout(args)
    perpective()
