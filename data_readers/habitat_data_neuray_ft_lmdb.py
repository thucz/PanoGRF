
from torch.utils.data import Dataset
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse

import numpy as np
import torch
import cv2
import random
from data_readers.create_rgb_dataset import RandomImageGenerator
from helpers import my_helpers
from helpers import my_torch_helpers

from dataset.database import M3DDatabase
from utils.imgs_info import build_imgs_info, imgs_info_to_torch
from utils.base_utils import get_coords_mask
from torchvision import transforms

import lmdb    



class HabitatImageGeneratorFT_LMDB(torch.utils.data.Dataset):

  default_cfg={
      'seq_len': 3, 
  }
  def __init__(self, args, split,
               vectorize=True,
               seed=0,
               full_width=512,
               full_height=256,
               seq_len=2,
               reference_idx=1,
               m3d_dist=1.0,
               use_rand=True,
               aug=False,
               ):
    self.cfg={**self.default_cfg,**args}
    self.split = split


    self.train = True

    self.full_width = full_width
    self.full_height = full_height
    self.depth_to_dist_cache = {}
    self.seq_len = seq_len
    self.reference_idx = reference_idx
    self.m3d_dist = m3d_dist
    self.use_rand = use_rand
    # self.num=99999999
    self.aug = aug
    print("Matterport3D distance:", m3d_dist)
    self.color_augmentation = True #not disable_color_augmentation
    self.LR_filp_augmentation = True#not disable_LR_filp_augmentation
    self.yaw_rotation_augmentation = True#not disable_yaw_rotation_augmentation
    if self.color_augmentation:
      try:
          self.brightness = (0.8, 1.2)
          self.contrast = (0.8, 1.2)
          self.saturation = (0.8, 1.2)
          self.hue = (-0.1, 0.1)
          self.color_aug= transforms.ColorJitter(
              self.brightness, self.contrast, self.saturation, self.hue)
      except TypeError:
          self.brightness = 0.2
          self.contrast = 0.2
          self.saturation = 0.2
          self.hue = 0.1
          self.color_aug = transforms.ColorJitter(
              self.brightness, self.contrast, self.saturation, self.hue)
    env_path = args["save_dir"]+'/lmdb_render_'+split+"_"+str(args["width"])+"x"+str(args["height"])+"_seq_len_"+str(args["seq_len"])+ \
      "_m3d_dist_"+str(args["m3d_dist"])
    
    if "freeview" in self.cfg and self.cfg["freeview"]:
      env_path += "_freeview_"+str(self.cfg["offset"][0])+","+str(self.cfg["offset"][1])+","+str(self.cfg["offset"][2])

    self.env = lmdb.open(env_path, create=False, subdir=True, readonly=True, lock=False)
    self.txn = self.env.begin(write=False)
    #self.env.close()在使用完数据记得close


  def __len__(self):
    return self.cfg["total_cnt"]


  def generate_coords_for_training(self, database, que_imgs_info):
    # if (database.database_name.startswith('real_estate') \
    #         or database.database_name.startswith('real_iconic') \
    #         or database.database_name.startswith('space')) and self.cfg['aug_pixel_center_sample']:
    #         que_mask_cur = np.zeros_like(que_imgs_info['masks'][0, 0]).astype(np.bool)
    #         h, w = que_mask_cur.shape
    #         center_ratio = 0.8
    #         begin_ratio = (1-center_ratio)/2
    #         hb, he = int(h*begin_ratio), int(h*(center_ratio+begin_ratio))
    #         wb, we = int(w*begin_ratio), int(w*(center_ratio+begin_ratio))
    #         que_mask_cur[hb:he,wb:we] = True
    #         coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], 0.9).reshape([1, -1, 2])
    # else:
    que_mask_cur = que_imgs_info['masks'][0,0]>0 #
    coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], 1.0).reshape([1,-1,2])
    return coords

  def read_lmdb(self, base_key, key_post, dtype):
    key = base_key+","+key_post
    buf = self.txn.get(key.encode('ascii'))
    data = np.frombuffer(buf, dtype=dtype)
    return data

  def __getitem__(self, idx):#index
    
    # Ignore the item and just generate an image
    base_key = str(idx)
    cube_width = self.full_height//2
    rgb_panos = self.read_lmdb(base_key, "rgb_panos", np.float32).reshape(self.seq_len, self.full_height, self.full_width, 3)
    depth_panos = self.read_lmdb(base_key, "depth_panos", np.float32).reshape(self.seq_len, self.full_height, self.full_width)    
    rots = self.read_lmdb(base_key, "rots", np.float32).reshape(self.seq_len, 3, 3)
    trans = self.read_lmdb(base_key, "trans", np.float32).reshape(self.seq_len, 3)

    # import ipdb;ipdb.set_trace()
    rgb_cubes = self.read_lmdb(base_key, "rgb_cubes", np.float32).reshape(self.seq_len, 6, cube_width, cube_width, 3)
    depth_cubes = self.read_lmdb(base_key, "depth_cubes", np.float32).reshape(self.seq_len, 6, cube_width, cube_width)    
    rots_cubes = self.read_lmdb(base_key, "rots_cubes", np.float32).reshape(self.seq_len, 6, 3, 3)
    trans_cubes = self.read_lmdb(base_key, "trans_cubes", np.float32).reshape(self.seq_len, 6, 3)

    data ={
      "rgb_panos": rgb_panos,
      "depth_panos": depth_panos,
      "rots": rots,
      "trans": trans,
      "rgb_cubes": rgb_cubes,
      "depth_cubes": depth_cubes,
      "rots_cubes": rots_cubes,
      "trans_cubes": trans_cubes
    }



    if self.split=="train":
      if self.aug:#todo aug
        #h, w, 3
        if len(data["rgb_panos"])==1:
          assert len(data["rgb_panos"])==1, "seq_len should be 1."

          rgb = data["rgb_panos"][0]
          gt_depth = data["depth_panos"][0]
          h, w, channels = rgb.shape

          if self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

          if self.LR_filp_augmentation and random.random() > 0.5:
              rgb = cv2.flip(rgb, 1)
              gt_depth = cv2.flip(gt_depth, 1)

          if self.color_augmentation and random.random() > 0.5:
              # import ipdb;ipdb.set_trace()
              aug_rgb = np.array(self.color_aug(transforms.ToPILImage()(np.uint8(rgb*255))))
              aug_rgb = aug_rgb * 1.0 / 255
          else:
              aug_rgb = rgb
          
          data["rgb_panos"] = aug_rgb[np.newaxis, ...].astype("float32")
          data["depth_panos"] = gt_depth[np.newaxis, ...]
        # else:
        #   # color augmentation
        #   color_aug = False
        #   if self.color_augmentation and random.random() > 0.5:  

        #     color_aug = True
        #     aug_gamma = random.uniform(0.9, 1.1)
        #     aug_brightness = random.uniform(0.75, 1.25)
        #     aug_colors = np.random.uniform(0.9, 1.1, size=3)
        #     for rgb_idx in range(len(data["rgb_panos"])):
        #       rgb = data["rgb_panos"][rgb_idx]
        #       # import ipdb;ipdb.set_trace()
        #       if color_aug:
        #         rgb = self.augment_image(rgb, aug_gamma, aug_brightness, aug_colors)
        #       # aug_rgb = np.array(self.color_aug(transforms.ToPILImage()(np.uint8(rgb*255))))
        #       # aug_rgb = aug_rgb * 1.0 / 255
        #       data["rgb_panos"][rgb_idx] = rgb #.astype("float32")
            
      #augmentation
      




    return data
  def augment_image(self, image, gamma, brightness, colors):
    # gamma augmentation
    image_aug = image ** gamma

    # brightness augmentation
    image_aug = image_aug * brightness
    # color augmentation
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)
    return image_aug

  def distance_to_zdepth_torch(self, distance_image):
    """Converts a depth (z-depth) image to a euclidean distance image.

    Args:
      depth_image: Equirectangular depth image as BxHxWx1 tensor.

    Returns: Equirectangular distance image.

    """
    batch_size, height, width, channels = distance_image.shape
    cache_key = "_".join((str(height), str(width)))
    self.cache_depth_to_dist(height, width)
    ratio = self.depth_to_dist_cache[cache_key]
    ratio = ratio[np.newaxis, :, :, np.newaxis]
    ratio = torch.tensor(ratio,
                         dtype=distance_image.dtype,
                         device=distance_image.device)
    zdepth_image = my_torch_helpers.safe_divide(distance_image, ratio)
    return zdepth_image

  def cache_depth_to_dist(self, height, width):
    """Caches a depth to dist ratio"""
    cache_key = "_".join((str(height), str(width)))
    if cache_key not in self.depth_to_dist_cache:
      cubemap_height = 256
      cubemap_width = 256
      # Distance to image plane
      theta, phi = np.meshgrid(
        (np.arange(width) + 0.5) * (2 * np.pi / width),
        (np.arange(height) + 0.5) * (np.pi / height))
      uvs, uv_sides = my_helpers.spherical_to_cubemap(theta.reshape(-1),
                                                      phi.reshape(-1))
      cubemap_uvs = uvs.reshape(height, width, 2)
      uv_int = np.stack(
        (cubemap_uvs[:, :, 0] * (cubemap_width - 1),
         cubemap_uvs[:, :, 1] *
         (cubemap_height - 1)),
        axis=-1)
      width_center = cubemap_width / 2 - 0.5
      height_center = cubemap_height / 2 - 0.5
      focal_len = (cubemap_height / 2) / np.tan(np.pi / 4)
      diag_dist = np.sqrt((uv_int[:, :, 0] - width_center) ** 2 +
                          (uv_int[:, :,
                           1] - height_center) ** 2 + focal_len ** 2)
      self.depth_to_dist_cache[cache_key] = diag_dist / focal_len

class FinetuningRendererDataset(Dataset):
    default_cfg={
        "database_name": "m3d",
        # "database_split": "val_all",
        # "m3d_idx": 0,
    }
    def __init__(self,cfg, is_train):
        self.cfg={**self.default_cfg,**cfg}
        self.is_train=is_train
        # self.train_ids, self.val_ids = get_database_split(parse_database_name(self.cfg['database_name']),self.cfg['database_split'])
        self.train_ids = [0, 2]
        self.val_ids = [1]

    def __getitem__(self, index):
        output={'index': index}
        return output

    def __len__(self):
        if self.is_train:
            return 99999999
        else:
            return len(self.val_ids)