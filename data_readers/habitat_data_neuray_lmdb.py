# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import os
import numpy as np
import torch
import cv2
from data_readers.create_rgb_dataset import RandomImageGenerator
from helpers import my_helpers
from helpers import my_torch_helpers

from dataset.database import M3DDatabase
from utils.imgs_info import build_imgs_info, imgs_info_to_torch, build_cube_imgs_info
from utils.base_utils import get_coords_mask
import lmdb

    


class HabitatImageGenerator_LMDB(torch.utils.data.Dataset):
  default_cfg={
      'seq_len': 3, 
      'train_database_types':['dtu_train','space','real_iconic','real_estate','gso'],
      # 'type2sample_weights': {'gso':20, 'dtu_train':20, 'real_iconic':20, 'space':10, 'real_estate':10},
      'val_database_name': 'nerf_synthetic/lego/black_800',
      'val_database_split_type': 'val',
      # 'min_wn': 8, #reference view window(numbers) (min_window_number, max_window_number)
      # 'max_wn': 9, #reference_view_window
      # 'ref_pad_interval': 16,# reference view padding
      'train_ray_num': 512, #
      'foreground_ratio': 1.0,
      'resolution_type': 'hr',
      # "use_consistent_depth_range": True,
      'use_depth_loss_for_all': False,
      "use_depth": False,
      "use_src_imgs": True,
      # "cost_volume_nn_num": 3,

      # "aug_gso_shrink_range_prob": 0.5,
      # "aug_depth_range_prob": 0.05,
      # 'aug_depth_range_min': 0.95,
      # 'aug_depth_range_max': 1.05,
      # "aug_use_depth_offset": True,
      # "aug_depth_offset_prob": 0.25,
      # "aug_depth_offset_region_min": 0.05,
      # "aug_depth_offset_region_max": 0.1,
      # 'aug_depth_offset_min': 0.5,
      # 'aug_depth_offset_max': 1.0,
      # 'aug_depth_offset_local': 0.1,
      # "aug_use_depth_small_offset": True,
      # "aug_use_global_noise": True,
      # "aug_global_noise_prob": 0.5,
      # "aug_depth_small_offset_prob": 0.5,
      # "aug_forward_crop_size": (400,600),
      # "aug_pixel_center_sample": False,
      # "aug_view_select_type": "easy",

      # "use_consistent_min_max": False,
      # "revise_depth_range": False,
  }

  def __init__(self, args, split,
               vectorize=True,
               seed=0,
               full_width=512,
               full_height=256,
               seq_len=2,
               reference_idx=1,
               m3d_dist=1.0,
               use_rand=True
               ):
    self.cfg={**self.default_cfg,**args}

    # self.worker_id = 0
    self.split = split
    # opts = ArgumentParser().parse(arg_str="")[0]
    # opts.train_data_path = (
    #   "./data_readers/scene_episodes/mp3d_train/dataset_one_ep_per_scene.json.gz"
    # )
    # opts.val_data_path = (
    #   "./data_readers/scene_episodes/mp3d_val/dataset_one_ep_per_scene.json.gz"
    # )
    # opts.test_data_path = (
    #   "./data_readers/scene_episodes/mp3d_test/dataset_one_ep_per_scene.json.gz"
    # )
    # opts.scenes_dir = "/group/30042/ozhengchen/"#ft_local/data_360/v1/tasks"  # this should store mp3d

    # self.opts = opts

    self.num_views = seq_len
    # self.vectorize = vectorize

    # self.image_generator = None

    # Part of hacky code to have train/val
    # self.episodes = None
    # self.restarted = True
    self.train = True

    # self.rng = np.random.RandomState(seed)
    # self.seed = opts.seed

    self.fixed_val_images = [None] * 32  # Keep 32 examples
    self.full_width = full_width
    self.full_height = full_height
    self.depth_to_dist_cache = {}
    self.seq_len = seq_len
    self.reference_idx = reference_idx
    self.m3d_dist = m3d_dist
    # self.use_rand = use_rand
    # self.num=99999999
    print("Matterport3D distance:", m3d_dist)

    # lmdb
    # if "use_lmdb" in self.cfg and self.cfg["use_lmdb"]:
    env_path = args["save_dir"]+'/lmdb_render_'+split+"_"+str(args["width"])+"x"+str(args["height"])+"_seq_len_"+str(args["seq_len"])+ \
      "_m3d_dist_"+str(args["m3d_dist"])
    self.env = lmdb.open(env_path, create=False, subdir=True, readonly=True, lock=False)
    self.txn = self.env.begin(write=False)
    #self.env.close()在使用完数据记得close

  def __len__(self):
    # return 2**31
    return self.cfg["total_cnt"]


  # def generate_coords_for_training(self, database, que_imgs_info):
  #   # if (database.database_name.startswith('real_estate') \
  #   #         or database.database_name.startswith('real_iconic') \
  #   #         or database.database_name.startswith('space')) and self.cfg['aug_pixel_center_sample']:
  #   #         que_mask_cur = np.zeros_like(que_imgs_info['masks'][0, 0]).astype(np.bool)
  #   #         h, w = que_mask_cur.shape
  #   #         center_ratio = 0.8
  #   #         begin_ratio = (1-center_ratio)/2
  #   #         hb, he = int(h*begin_ratio), int(h*(center_ratio+begin_ratio))
  #   #         wb, we = int(w*begin_ratio), int(w*(center_ratio+begin_ratio))
  #   #         que_mask_cur[hb:he,wb:we] = True
  #   #         coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], 0.9).reshape([1, -1, 2])
  #   # else:
    
  #   return coords

  #env
  def read_lmdb(self, base_key, key_post, dtype):
    key = base_key+","+key_post
    buf = self.txn.get(key.encode('ascii'))
    data = np.frombuffer(buf, dtype=dtype)
    return data

  def __getitem__(self, idx):#index
    # Ignore the item and just generate an image
    # data = self.image_generator.get_sample(item, self.num_views, self.train)
    base_key = str(idx)

    rgb_panos = self.read_lmdb(base_key, "rgb_panos", np.float32).reshape(self.seq_len, self.full_height, self.full_width, 3)
    depth_panos = self.read_lmdb(base_key, "depth_panos", np.float32).reshape(self.seq_len, self.full_height, self.full_width)    
    rots = self.read_lmdb(base_key, "rots", np.float32).reshape(self.seq_len, 3, 3)
    trans = self.read_lmdb(base_key, "trans", np.float32).reshape(self.seq_len, 3)
    
    cube_width = self.full_height//2
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
    
    database = M3DDatabase(self.cfg, data)
    
    # data["rgb_panos"]
    # self.images  = data["rgb_panos"]#.to(args.device)
    # self.depths = data["depth_panos"]#.to(args.device)
    # self.rots = data["rots"]#.to(args.device)
    # self.trans = data["trans"]#.to(args.device)
    ids_all = [0, 2]    
    # if self.cfg["debug"]:
    src_dict={
      0: 2,
      2: 0,
    }
    # import ipdb;ipdb.set_trace()
    if "render_cubes" in self.cfg and self.cfg["render_cubes"] and self.split == "train":
      # if self.split == "train":
      render_ids = list(range(6, 6*2))
      que_id = render_ids[np.random.randint(0, len(render_ids))]
      ref_ids_all = [0, 2]
      np.random.shuffle(ref_ids_all) #list(set([0, 2]) - set([que_id]))
      src_ids = [src_dict[ref_id] for ref_id in ref_ids_all]
        # print("que_id, ref_ids:", que_id, ref_ids_all)        
      que_imgs_info = build_cube_imgs_info(database, [que_id], has_depth=True)
      que_mask_cur = que_imgs_info['cube_masks'][0,0]>0 #
      # print("que_mask_cur.shape:", que_mask_cur.shape)
      # import ipdb;ipdb.set_trace()
      # 1, 512, 2
      coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], self.cfg['foreground_ratio']).reshape([1,-1,2])
      que_imgs_info['coords'] = coords
    
    else:
      if self.split == "train":
        que_id = 1
        ref_ids_all = [0, 2]
        np.random.shuffle(ref_ids_all) #list(set([0, 2]) - set([que_id]))
        src_ids = [src_dict[ref_id] for ref_id in ref_ids_all]
        print("que_id, ref_ids:", que_id, ref_ids_all)        
      else:
        que_id = 1
        ref_ids_all = [0, 2]#list(set([0, 2]) - set([que_id]))
        src_ids = [src_dict[ref_id] for ref_id in ref_ids_all]
      que_imgs_info = build_imgs_info(database, [que_id], has_depth=True)

      if self.split=="train":
        que_mask_cur = que_imgs_info['masks'][0,0]>0 #
        # print("que_mask_cur.shape:", que_mask_cur.shape)
        # import ipdb;ipdb.set_trace()
        # 1, 512, 2
        coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], self.cfg['foreground_ratio']).reshape([1,-1,2])
      else:
        qn, _, hn, wn = que_imgs_info['imgs'].shape
        coords = np.stack(np.meshgrid(np.arange(wn),np.arange(hn)),-1)
        coords = coords.reshape([1,-1,2]).astype(np.float32)
      que_imgs_info['coords'] = coords
    
    # src_imgs_info used in construction of cost volume
    # ref_imgs_info, ref_cv_idx, ref_real_idx = build_src_imgs_info_select(database,ref_ids,ref_id)    
    #todo: ref_cv_idx, ref_real_idx
    ref_imgs_info = build_imgs_info(database, ref_ids_all, has_depth=True)#?  
    if self.split=="train":
      # data augmentation
      depth_range_all = np.concatenate([ref_imgs_info['depth_range'],que_imgs_info['depth_range']],0)
      depth_all, mask_all = None, None  
      if self.cfg['use_depth_loss']:# and self.cfg['use_depth']:
        ref_imgs_info['true_depth'] = ref_imgs_info['depth']
      # generate coords
    
      
    

    # don't feed depth to gpu
    # import ipdb;ipdb.set_trace()
      # if not self.cfg['use_depth']:
      #   if 'depth' in ref_imgs_info: ref_imgs_info.pop('depth')
      #   if 'depth' in que_imgs_info: que_imgs_info.pop('depth')
      #   if 'true_depth' in ref_imgs_info: ref_imgs_info.pop('true_depth')

    assert len(ref_imgs_info["imgs"]) <= 2, "Only ref_num<=2 is supported now"
    src_imgs_info = build_imgs_info(database, src_ids)#?

    ref_imgs_info = imgs_info_to_torch(ref_imgs_info)
    que_imgs_info = imgs_info_to_torch(que_imgs_info)

    outputs = {'ref_imgs_info': ref_imgs_info, 'que_imgs_info': que_imgs_info}
    if self.cfg['use_src_imgs']: outputs['src_imgs_info'] = imgs_info_to_torch(src_imgs_info)
    
    return outputs

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