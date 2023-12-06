
import abc
import glob
import json
import os
import random
import re
from pathlib import Path
import torch
import cv2
import numpy as np
from skimage.io import imread, imsave

from asset import LLFF_ROOT, nerf_syn_val_ids, NERF_SYN_ROOT
# from colmap.read_write_dense import read_array
# from colmap.read_write_model import read_cameras_binary, read_images_binary, read_points3d_binary
from utils.base_utils import color_map_backward 

# , resize_img, read_pickle, project_points, \
    # save_pickle, transform_points_Rt, pose_inverse, downsample_gaussian_blur, 
from PIL import Image

# from utils.llff_utils import load_llff_data
# from utils.real_estate_utils import parse_pose_file, unnormalize_intrinsics
# from utils.space_dataset_utils import ReadScene

# data_dir="/group/30042/ozhengchen/ft_local/neuray_data"

class BaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass


    @abc.abstractmethod
    def get_pose(self, img_id):
        pass

    @abc.abstractmethod
    def get_img_ids(self,check_depth_exist=False):
        pass

    @abc.abstractmethod
    def get_bbox(self, img_id):
        pass

    @abc.abstractmethod
    def get_depth(self,img_id):
        pass

    @abc.abstractmethod
    def get_mask(self,img_id):
        pass

    @abc.abstractmethod
    def get_depth_range(self,img_id):
        pass

def get_poses(rots, trans):
    # import ipdb;ipdb.set_trace()
    seq_len = rots.shape[0]
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
    w2c_list = []
    c2w_list = []
    
    for idx in range(seq_len):
        rot = rots[idx, ...]
        tr = trans[idx, ...]
        pose = np.concatenate([np.concatenate([rot, tr[..., np.newaxis]], axis=1), bottom], axis=0)
        # pose = torch.from_numpy(np.concatenate([rot, tr[..., np.newaxis]], axis=1))#w2c        
        w2c = pose[:3, :]
        c2w = np.linalg.inv(pose)[:3, :] #c2w
        # pose = pose[:3, :]
        # pose_list.append(pose)
        w2c_list.append(w2c)
        c2w_list.append(c2w)

    w2c_list = np.stack(w2c_list)
    c2w_list = np.stack(c2w_list)
    return w2c_list, c2w_list


# poses = get_poses(rots, trans)

class M3DDatabase:
    def __init__(self, args, data):#data
        # super(M3DDatabase, self).__init__()
        self.database_name = args["dataset_name"]
        # self.images, poses, self.range_dict, self.render_poses, self.test_img_id
        
        # self.images, poses, self.test_img_id = parse(data)
        self.cfg = args
        self.images  = data["rgb_panos"]#.to(args.device)
        self.depths = data["depth_panos"]#.to(args.device)
        self.rots = data["rots"]#.to(args.device)
        self.trans = data["trans"]#.to(args.device)
        if "render_cubes" in args:
            # import ipdb;ipdb.set_trace()
            cube_width = args["height"]//2 #256//2
            # import ipdb;ipdb.set_trace()
            seq_len = args['seq_len']
            self.rgb_cubes = data["rgb_cubes"].reshape(6*seq_len, cube_width, cube_width, 3) # 1, 3, 6, 256, 256, 3        
            self.depth_cubes = data["depth_cubes"].reshape(6*seq_len, cube_width, cube_width) # 1, 3, 6, 256, 256
            self.trans_cubes = data["trans_cubes"].reshape(6*seq_len, 3) #1, 3, 6, 3
            self.rots_cubes = data["rots_cubes"].reshape(6*seq_len, 3, 3) 
            
            # import ipdb;ipdb.set_trace()
            H, W = self.rgb_cubes.shape[1:3] #[2:4]
            # intrinsic parameters
            FOV=90
            f = 0.5 * W * 1 / np.tan(0.5 * FOV / 180.0 * np.pi) #
            cx = (W - 1) / 2.0
            cy = (H - 1) / 2.0
            self.K = np.array([
                    [f, 0, cx],
                    [0, f, cy],
                    [0, 0,  1],
                ], np.float32)
            # print("K:", self.K)
            # import ipdb;ipdb.set_trace()
            self.w2c_cubes, self.c2w_cubes = get_poses(self.rots_cubes, self.trans_cubes)
            self.cube_img_ids = [k for k in range(len(self.rgb_cubes))]
        self.w2c, self.c2w = get_poses(self.rots, self.trans)
        self.img_ids = [k for k in range(len(self.images))]
        # test_views = 
        if "test_views" in args:
            self.test_img_ids = args["test_views"]
        else:
            self.test_img_ids = [1]
        self.train_img_ids = [k for k in self.img_ids if k not in self.test_img_ids]

        self.range_dict=np.asarray([args["min_depth"], args["max_depth"]])
        self.depth_img_ids = list(range(len(self.depths)))
        
        
    def get_image(self, img_id):
        return self.images[int(img_id)].copy()

    def get_cube_image(self, img_id):
        return self.rgb_cubes[int(img_id)].copy()

    def get_rots(self, img_id):
        return self.rots[int(img_id)].copy()
    
    def get_cube_rots(self, img_id):
        return self.rots_cubes[int(img_id)].copy()

    def get_trans(self, img_id):
        return self.trans[int(img_id)].copy()

    def get_cube_trans(self, img_id):
        return self.trans_cubes[int(img_id)].copy()

    def get_c2w(self, img_id):
        return self.c2w[int(img_id)].copy()

    def get_w2c(self, img_id):
        return self.w2c[int(img_id)].copy()

    def get_cube_c2w(self, img_id):
        return self.c2w_cubes[int(img_id)].copy()

    def get_K(self, img_id):
        return self.K        

    def get_cube_pose(self, img_id):
        return self.w2c_cubes[int(img_id)].copy()    

    def get_cube_w2c(self, img_id):
        return self.w2c_cubes[int(img_id)].copy()

    def get_img_ids(self,check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.img_ids
    
    def get_cube_img_ids(self,check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.cube_img_ids
    

    def get_bbox(self, img_id):
        raise NotImplementedError
    def _depth_existence(self, img_id):
        raise NotImplementedError    
    def get_depth(self, img_id):
        return self.depths[img_id].copy()

    def get_mask(self, img_id):
        h, w = self.get_image(img_id).shape[:2]
        return np.ones([h,w],dtype=np.float32).copy()
    
    def get_cube_mask(self, img_id):
        h, w = self.get_cube_image(img_id).shape[:2]
        return np.ones([h,w],dtype=np.float32).copy()

    def get_cube_depth(self, img_id):
        return self.depth_cubes[img_id].copy()

    def get_cube_mask(self, img_id):
        h, w = self.get_cube_image(img_id).shape[:2]
        return np.ones([h,w],dtype=np.float32).copy()

    def get_depth_range(self,img_id):
        return self.range_dict.copy()

    def get_cube_interpolate_render_poses(self, inter_num=5, cube_id=4):
        def interpolate_views(n_views_add, start_pose, end_pose):
            delta = (end_pose - start_pose)/(n_views_add+1)#
            new_poses_add = []
            for i in range(n_views_add):
                pose_add = start_pose + delta*(i+1)
                new_poses_add.append(pose_add)
            return new_poses_add

        def interpolate_render_poses(inter_img_ids, view_num):    
            poses = [self.get_cube_w2c(str(img_id)) for img_id in inter_img_ids]
            #在已有的poses中进行插值，假设输入的poses按照固定的拍摄顺序
            add_poses_len = view_num - len(poses)#58
            add = add_poses_len // (len(poses)-1)#58
            rest = add_poses_len % (len(poses)-1)#0
            print('add, rest:', add, rest)
            new_poses = []
            #poses[i]->poses[i+1]
            for i in range(len(poses)-1):
                # i, i+1
                # interpolate views
                if i < rest:
                #     #poses[i]和poses[i+1]的pose
                    add_poses = interpolate_views(add+1, poses[i], poses[i+1])
                else:
                    add_poses = interpolate_views(add, poses[i], poses[i+1])
                new_poses.append(poses[i])
                new_poses+=add_poses
            new_poses.append(poses[-1])#last pose
            new_poses = np.array(new_poses)
            return new_poses

        # cube_id+6*[0, 2]
        pano_img_ids = [0, 2]
        cube_img_ids = [pano_id * 6 + cube_id for pano_id in pano_img_ids]
        que_poses = interpolate_render_poses(cube_img_ids, inter_num)
        return que_poses


class ReplicaDatabase:
    def __init__(self, args):#data
        self.args = args
        data_dir = args["data_dir"]
        data = np.load(data_dir)
        # pano_path=processed_data["panos_path"], panos = processed_data["panos"], poses=processed_data["poses"])
        data_name="office"

        #office:
        if "data_name" in args and args["data_name"] == "office":
            #5-12

            self.images = data["panos"][5:12:3] #N, H, W, 3
            self.poses = data["poses"][5:12:3] #N, 4, 4
            self.w2c = data["w2c"][5:12:3]
            self.c2w = data["c2w"][5:12:3]
            self.trans = data["trans"][5:12:3]
            self.rots = data["rots"][5:12:3]
        else:
            self.images = data["panos"][:3] #N, H, W, 3
            self.poses = data["poses"][:3] #N, 4, 4
            self.w2c = data["w2c"][:3]
            self.c2w = data["c2w"][:3]
            self.trans = data["trans"][:3]
            self.rots = data["rots"][:3]


        # self.poses = data["poses"]
        # self.poses = 

        #suppossing w2c
        # self.w2c = self.poses.copy()
        # self.rots, self.trans, self.w2c, self.c2w = get_replica_poses(self.poses)

        # self.images  = data["rgb_panos"]#.to(args.device)
        # self.depths = data["depth_panos"]#.to(args.device)

        # self.rots = data["rots"]#.to(args.device)
        # self.trans = data["trans"]#.to(args.device)
                
        self.w2c, self.c2w = get_poses(self.rots, self.trans)        
        self.img_ids = [k for k in range(len(self.images))]
        # self.test_img_ids=[1]
        # self.train_img_ids=[k for k in self.img_ids if k not in self.test_img_ids]
        self.range_dict=np.asarray([args["min_depth"], args["max_depth"]])
        # self.depth_img_ids = list(range(len(self.depths)))

    def get_image(self, img_id):
        res = cv2.resize(self.images[int(img_id)], (self.args["width"], self.args["height"]), cv2.INTER_LINEAR)
        # # return self.images[int(img_id)].copy()
        # cv2.imwrite("res.jpg", np.uint8(res*255))
        return res.copy()


    def get_rots(self, img_id):
        return self.rots[int(img_id)].copy()

    def get_trans(self, img_id):
        return self.trans[int(img_id)].copy()

    def get_c2w(self, img_id):
        return self.c2w[int(img_id)].copy()

    def get_w2c(self, img_id):
        return self.w2c[int(img_id)].copy()

    def get_img_ids(self,check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.img_ids

    
    def get_depth(self, img_id):
        return self.depths[img_id].copy()
        
    def get_mask(self, img_id):
        h, w = self.get_image(img_id).shape[:2]
        # print("get mask h, w:", h, w)
        return np.ones([h,w],dtype=np.float32).copy()

    def get_depth_range(self,img_id):
        return self.range_dict.copy()

class ResidentialDatabase:
    def __init__(self, args, data):#data
        # super(M3DDatabase, self).__init__()
        self.database_name = args["dataset_name"]
        # self.images, poses, self.range_dict, self.render_poses, self.test_img_id
        
        # self.images, poses, self.test_img_id = parse(data)
        self.images  = data["rgbs"].permute(0, 2, 3, 1).data.cpu().numpy()#.to(args.device)
        self.c2w = data["c2w"]#.data.cpu().numpy()
        # self.depths = data["depth_panos"]#.to(args.device)
        if "render_cubes" in args:
            cube_width = args["height"]//2 #256//2
            self.rgb_cubes = data["cube_rgbs"].squeeze().permute(0, 1, 3, 4, 2).reshape(18, cube_width, cube_width, 3).data.cpu().numpy()
            self.c2w_cubes = data["cube_c2w"].reshape(18, 4, 4)#[:, :3, :]#.data.cpu().numpy()
            # self.rgb_cubes = data["rgb_cubes"].reshape(18, cube_width, cube_width, 3)#1,3,6, 256, 256, 3        
            # self.depth_cubes = data["depth_cubes"].reshape(18, cube_width, cube_width) # 1,3,6, 256, 256
            # self.trans_cubes = data["trans_cubes"].reshape(18, 3)#1, 3, 6, 3
            # self.rots_cubes = data["rots_cubes"].reshape(18, 3, 3)        
            H, W = self.rgb_cubes.shape[1:3] #[2:4]
            # intrinsic parameters
            FOV=90
            f = 0.5 * W * 1 / np.tan(0.5 * FOV / 180.0 * np.pi) #
            cx = (W - 1) / 2.0
            cy = (H - 1) / 2.0
            self.K = np.array([
                    [f, 0, cx],
                    [0, f, cy],
                    [0, 0,  1],
                ], np.float32)
            # print("K:", self.K)
            # self.w2c_cubes, self.c2w_cubes = get_poses(self.rots_cubes, self.trans_cubes)
            self.w2c_cubes = torch.linalg.inv(self.c2w_cubes)
            self.w2c_cubes = self.w2c_cubes[:, :3, :].data.cpu().numpy()
            self.c2w_cubes = self.c2w_cubes[:, :3, :].data.cpu().numpy()
            self.cube_img_ids = [k for k in range(len(self.rgb_cubes))]        
        # self.w2c, self.c2w = get_poses(self.rots, self.trans)        
        self.w2c = torch.linalg.inv(self.c2w)
        self.w2c = self.w2c[:, :3, :].data.cpu().numpy()
        self.c2w = self.c2w[:, :3, :].data.cpu().numpy()
        self.rots = self.w2c[:, :3, :3]#data["rots"]#.to(args.device)
        self.trans = self.w2c[:, :3, 3]#data["trans"]#.to(args.device)
        self.img_ids = [k for k in range(len(self.images))]
        self.test_img_ids=[1]
        self.train_img_ids=[k for k in self.img_ids if k not in self.test_img_ids]

        # self.range_dict={str(k):np.asarray(self.range_dict[k],np.float32) for k in range(len(self.range_dict))}
        self.range_dict=np.asarray([args["min_depth"], args["max_depth"]])
        # self.depth_img_ids = list(range(len(self.depths)))

    def get_image(self, img_id):
        return self.images[int(img_id)].copy()

    def get_cube_image(self, img_id):
        return self.rgb_cubes[int(img_id)].copy()

    def get_rots(self, img_id):
        return self.rots[int(img_id)].copy()
    
    # def get_cube_rots(self, img_id):
    #     return self.rots_cubes[int(img_id)].copy()

    def get_trans(self, img_id):
        return self.trans[int(img_id)].copy()

    # def get_cube_trans(self, img_id):
    #     return self.trans_cubes[int(img_id)].copy()

    def get_c2w(self, img_id):
        return self.c2w[int(img_id)].copy()

    def get_w2c(self, img_id):
        return self.w2c[int(img_id)].copy()

    def get_cube_c2w(self, img_id):
        return self.c2w_cubes[int(img_id)].copy()
    
    def get_K(self, img_id):
        return self.K        

    def get_cube_pose(self, img_id):
        return self.w2c_cubes[int(img_id)].copy()    

    def get_cube_w2c(self, img_id):
        return self.w2c_cubes[int(img_id)].copy()

    def get_img_ids(self,check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.img_ids
    
    def get_cube_img_ids(self,check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.cube_img_ids
    

    def get_bbox(self, img_id):
        raise NotImplementedError
    def _depth_existence(self, img_id):
        raise NotImplementedError    
    # def get_depth(self, img_id):
    #     return self.depths[img_id].copy()

    def get_mask(self, img_id):
        h, w = self.get_image(img_id).shape[:2]
        # print("get mask h, w:", h, w)
        return np.ones([h,w],dtype=np.float32).copy()
    
    def get_cube_mask(self, img_id):
        h, w = self.get_cube_image(img_id).shape[:2]
        # print("get mask h, w:", h, w)
        return np.ones([h,w],dtype=np.float32).copy()

    # def get_cube_depth(self, img_id):
    #     return self.depth_cubes[img_id].copy()

    def get_cube_mask(self, img_id):
        h, w = self.get_cube_image(img_id).shape[:2]
        # print("get mask h, w:", h, w)
        return np.ones([h,w],dtype=np.float32).copy()

    def get_depth_range(self,img_id):
        return self.range_dict.copy()

    
    def get_cube_interpolate_render_poses(self, inter_num=5, cube_id=4):
        def interpolate_views(n_views_add, start_pose, end_pose):
            #3x4
            #R,T
            delta = (end_pose - start_pose)/(n_views_add+1)#
            new_poses_add = []
            for i in range(n_views_add):
                pose_add = start_pose + delta*(i+1)
                new_poses_add.append(pose_add)
            return new_poses_add

        def interpolate_render_poses(inter_img_ids, view_num):    
            poses = [self.get_cube_w2c(str(img_id)) for img_id in inter_img_ids]
            #在已有的poses中进行插值，假设输入的poses按照固定的拍摄顺序
            add_poses_len = view_num - len(poses)#58
            add = add_poses_len // (len(poses)-1)#58
            rest = add_poses_len % (len(poses)-1)#0
            print('add, rest:', add, rest)
            new_poses = []
            #poses[i]->poses[i+1]
            for i in range(len(poses)-1):
                # i, i+1
                # interpolate views
                if i < rest:
                #     #poses[i]和poses[i+1]的pose
                    add_poses = interpolate_views(add+1, poses[i], poses[i+1])
                else:
                    add_poses = interpolate_views(add, poses[i], poses[i+1])
                new_poses.append(poses[i])
                new_poses+=add_poses
            new_poses.append(poses[-1])#last pose
            new_poses = np.array(new_poses)
            return new_poses

        # cube_id+6*[0, 2]
        pano_img_ids = [0, 2]
        cube_img_ids = [pano_id * 6 + cube_id for pano_id in pano_img_ids]
        que_poses = interpolate_render_poses(cube_img_ids, inter_num)
        return que_poses



class CoffeeAreaDatabase:
    def __init__(self, args, data):#data
        # super(M3DDatabase, self).__init__()
        self.database_name = args["dataset_name"]
        # self.images, poses, self.range_dict, self.render_poses, self.test_img_id
        
        # self.images, poses, self.test_img_id = parse(data)
        self.images  = data["rgbs"].permute(0, 2, 3, 1).data.cpu().numpy()#.to(args.device)
        # import ipdb;ipdb.set_trace()
        self.c2w = data["c2w"]#.data.cpu().numpy()
        # self.depths = data["depth_panos"]#.to(args.device)
        if "render_cubes" in args:
            cube_width = args["height"]//2 #256//2
            # import ipdb;ipdb.set_trace()
            # import ipdb;ipdb.set_trace()
            self.rgb_cubes = data["cube_rgbs"].squeeze().permute(0, 1, 3, 4, 2).reshape(18, cube_width, cube_width, 3).data.cpu().numpy()
            self.c2w_cubes = data["cube_c2w"].reshape(18, 4, 4)#[:, :3, :]#.data.cpu().numpy()
            # self.rgb_cubes = data["rgb_cubes"].reshape(18, cube_width, cube_width, 3)#1,3,6, 256, 256, 3        
            # self.depth_cubes = data["depth_cubes"].reshape(18, cube_width, cube_width) # 1,3,6, 256, 256
            # self.trans_cubes = data["trans_cubes"].reshape(18, 3)#1, 3, 6, 3
            # self.rots_cubes = data["rots_cubes"].reshape(18, 3, 3)        
            # import ipdb;ipdb.set_trace()
            H, W = self.rgb_cubes.shape[1:3] #[2:4]
            # intrinsic parameters
            FOV=90
            f = 0.5 * W * 1 / np.tan(0.5 * FOV / 180.0 * np.pi) #
            cx = (W - 1) / 2.0
            cy = (H - 1) / 2.0
            self.K = np.array([
                    [f, 0, cx],
                    [0, f, cy],
                    [0, 0,  1],
                ], np.float32)
            # print("K:", self.K)
            # import ipdb;ipdb.set_trace()
            # self.w2c_cubes, self.c2w_cubes = get_poses(self.rots_cubes, self.trans_cubes)
            self.w2c_cubes = torch.linalg.inv(self.c2w_cubes)
            self.w2c_cubes = self.w2c_cubes[:, :3, :].data.cpu().numpy()
            self.c2w_cubes = self.c2w_cubes[:, :3, :].data.cpu().numpy()
            self.cube_img_ids = [k for k in range(len(self.rgb_cubes))]
        # import ipdb;ipdb.set_trace()
        # self.w2c, self.c2w = get_poses(self.rots, self.trans)
        self.w2c = torch.linalg.inv(self.c2w)
        self.w2c = self.w2c[:, :3, :].data.cpu().numpy()
        self.c2w = self.c2w[:, :3, :].data.cpu().numpy()
        self.rots = self.w2c[:, :3, :3]#data["rots"]#.to(args.device)
        self.trans = self.w2c[:, :3, 3]#data["trans"]#.to(args.device)

        self.img_ids = [k for k in range(len(self.images))]

        self.test_img_ids=[1]
        self.train_img_ids=[k for k in self.img_ids if k not in self.test_img_ids]

        # self.range_dict={str(k):np.asarray(self.range_dict[k],np.float32) for k in range(len(self.range_dict))}
        self.range_dict=np.asarray([args["min_depth"], args["max_depth"]])
        # self.depth_img_ids = list(range(len(self.depths)))

    def get_image(self, img_id):
        return self.images[int(img_id)].copy()

    def get_cube_image(self, img_id):
        return self.rgb_cubes[int(img_id)].copy()

    def get_rots(self, img_id):
        return self.rots[int(img_id)].copy()
    
    # def get_cube_rots(self, img_id):
    #     return self.rots_cubes[int(img_id)].copy()

    def get_trans(self, img_id):
        return self.trans[int(img_id)].copy()

    # def get_cube_trans(self, img_id):
    #     return self.trans_cubes[int(img_id)].copy()

    def get_c2w(self, img_id):
        return self.c2w[int(img_id)].copy()

    def get_w2c(self, img_id):
        return self.w2c[int(img_id)].copy()

    def get_cube_c2w(self, img_id):
        return self.c2w_cubes[int(img_id)].copy()
    
    def get_K(self, img_id):
        return self.K        

    def get_cube_pose(self, img_id):
        return self.w2c_cubes[int(img_id)].copy()    

    def get_cube_w2c(self, img_id):
        return self.w2c_cubes[int(img_id)].copy()

    def get_img_ids(self,check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.img_ids
    
    def get_cube_img_ids(self,check_depth_exist=False):
        if check_depth_exist:
            return self.depth_img_ids
        return self.cube_img_ids
    

    def get_bbox(self, img_id):
        raise NotImplementedError
    def _depth_existence(self, img_id):
        raise NotImplementedError    
    # def get_depth(self, img_id):
    #     return self.depths[img_id].copy()

    def get_mask(self, img_id):
        h, w = self.get_image(img_id).shape[:2]
        # print("get mask h, w:", h, w)
        return np.ones([h,w],dtype=np.float32).copy()
    
    def get_cube_mask(self, img_id):
        h, w = self.get_cube_image(img_id).shape[:2]
        # print("get mask h, w:", h, w)
        return np.ones([h,w],dtype=np.float32).copy()

    # def get_cube_depth(self, img_id):
    #     return self.depth_cubes[img_id].copy()

    def get_cube_mask(self, img_id):
        h, w = self.get_cube_image(img_id).shape[:2]
        # print("get mask h, w:", h, w)
        return np.ones([h,w],dtype=np.float32).copy()

    def get_depth_range(self,img_id):
        return self.range_dict.copy()

    
    def get_cube_interpolate_render_poses(self, inter_num=5, cube_id=4):
        def interpolate_views(n_views_add, start_pose, end_pose):
            #3x4
            #R,T
            delta = (end_pose - start_pose)/(n_views_add+1)#
            new_poses_add = []
            for i in range(n_views_add):
                pose_add = start_pose + delta*(i+1)
                new_poses_add.append(pose_add)
            return new_poses_add

        def interpolate_render_poses(inter_img_ids, view_num):    
            poses = [self.get_cube_w2c(str(img_id)) for img_id in inter_img_ids]
            #在已有的poses中进行插值，假设输入的poses按照固定的拍摄顺序
            add_poses_len = view_num - len(poses)#58
            add = add_poses_len // (len(poses)-1)#58
            rest = add_poses_len % (len(poses)-1)#0
            print('add, rest:', add, rest)
            new_poses = []
            #poses[i]->poses[i+1]
            for i in range(len(poses)-1):
                # i, i+1
                # interpolate views
                if i < rest:
                #     #poses[i]和poses[i+1]的pose
                    add_poses = interpolate_views(add+1, poses[i], poses[i+1])
                else:
                    add_poses = interpolate_views(add, poses[i], poses[i+1])
                new_poses.append(poses[i])
                new_poses+=add_poses
            new_poses.append(poses[-1])#last pose
            new_poses = np.array(new_poses)
            return new_poses

        # cube_id+6*[0, 2]
        pano_img_ids = [0, 2]
        cube_img_ids = [pano_id * 6 + cube_id for pano_id in pano_img_ids]
        que_poses = interpolate_render_poses(cube_img_ids, inter_num)
        return que_poses


def get_database_split(database: BaseDatabase, split_type='val'):
    database_name = database.database_name
    # if split_type.startswith('val'):
    #     splits = split_type.split('_')
    #     depth_valid = not(len(splits)>1 and splits[1]=='all')
    #     if database_name.startswith('nerf_synthetic'):
    #         train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id.startswith('tr')]
    #         val_ids = nerf_syn_val_ids
    #     elif database_name.startswith('llff'):
    #         val_ids = database.get_img_ids()[::8]
    #         train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id not in val_ids]
    #     elif database_name.startswith('dtu_test'):
    #         val_ids = database.get_img_ids()[3:-3:8]
    #         train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id not in val_ids]
    #     else:
    #         raise NotImplementedError
    if split_type.startswith('test'):
        splits = split_type.split('_')
        depth_valid = not(len(splits)>1 and splits[1]=='all')
        if database_name.startswith('m3d'):
            train_ids = [0, 2] #[img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id.startswith('tr')]
            val_ids = [1]
            

        
        elif database_name.startswith('residential'):
            train_ids = [0, 2] #[img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id.startswith('tr')]
            val_ids = [1]
        elif database_name.startswith('CoffeeArea'):
            train_ids = [0, 2] #[img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id.startswith('tr')]
            val_ids = [1]

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return train_ids, val_ids
def get_database_split_mv(database: BaseDatabase, split_type='val'):
    database_name = database.database_name
    # if split_type.startswith('val'):
    #     splits = split_type.split('_')
    #     depth_valid = not(len(splits)>1 and splits[1]=='all')
    #     if database_name.startswith('nerf_synthetic'):
    #         train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id.startswith('tr')]
    #         val_ids = nerf_syn_val_ids
    #     elif database_name.startswith('llff'):
    #         val_ids = database.get_img_ids()[::8]
    #         train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id not in val_ids]
    #     elif database_name.startswith('dtu_test'):
    #         val_ids = database.get_img_ids()[3:-3:8]
    #         train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id not in val_ids]
    #     else:
    #         raise NotImplementedError
    if split_type.startswith('test'):
        splits = split_type.split('_')
        depth_valid = not(len(splits)>1 and splits[1]=='all')
        if database_name.startswith('m3d'):
            test_views = database.cfg['test_views']
            train_ids = list(range(database.cfg['reference_idx']))   #   list(set(range(database.cfg['seq_len'])) - set(test_views))     #[0, 2] #[img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id.startswith('tr')]
            val_ids = test_views
            # import ipdb;ipdb.set_trace()
        # elif database_name.startswith('residential'):
        #     train_ids = [0, 2] #[img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id.startswith('tr')]
        #     val_ids = [1]
        # elif database_name.startswith('CoffeeArea'):
        #     train_ids = [0, 2] #[img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id.startswith('tr')]
        #     val_ids = [1]

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return train_ids, val_ids