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
from utils.imgs_info import build_imgs_info, imgs_info_to_torch
from utils.base_utils import get_coords_mask


# , random_crop, random_flip, pad_imgs_info, imgs_info_slice, \
    

class ArgumentParser:
  def __init__(self):
    self.parser = argparse.ArgumentParser(
      description="self-supervised view synthesis"
    )
    self.add_data_parameters()
    self.add_train_parameters()
    self.add_model_parameters()

  def add_model_parameters(self):
    model_params = self.parser.add_argument_group("model")
    model_params.add_argument(
      "--model_type",
      type=str,
      default="zbuffer_pts",
      choices=(
        "zbuffer_pts",
        "deepvoxels",
        "viewappearance",
        "tatarchenko",
      ),
      help='Model to be used.'
    )
    model_params.add_argument(
      "--refine_model_type", type=str, default="unet",
      help="Model to be used for the refinement network and the feature encoder."
    )
    model_params.add_argument(
      "--accumulation",
      type=str,
      default="wsum",
      choices=("wsum", "wsumnorm", "alphacomposite"),
      help="Method for accumulating points in the z-buffer. Three choices: wsum (weighted sum), wsumnorm (normalised weighted sum), alpha composite (alpha compositing)"
    )

    model_params.add_argument(
      "--depth_predictor_type",
      type=str,
      default="unet",
      choices=("unet", "hourglass", "true_hourglass"),
      help='Model for predicting depth'
    )
    model_params.add_argument(
      "--splatter",
      type=str,
      default="xyblending",
      choices=("xyblending"),
    )
    model_params.add_argument("--rad_pow", type=int, default=2,
                              help='Exponent to raise the radius to when computing distance (default is euclidean, when rad_pow=2). ')
    model_params.add_argument("--num_views", type=int, default=2,
                              help='Number of views considered per input image (inlcluding input), we only use num_views=2 (1 target view).')
    model_params.add_argument(
      "--crop_size",
      type=int,
      default=256,
      help="Crop to the width of crop_size (after initially scaling the images to load_size.)",
    )
    model_params.add_argument(
      "--aspect_ratio",
      type=float,
      default=1.0,
      help="The ratio width/height. The final height of the load image will be crop_size/aspect_ratio",
    )
    model_params.add_argument(
      "--norm_D",
      type=str,
      default="spectralinstance",
      help="instance normalization or batch normalization",
    )
    model_params.add_argument(
      "--noise", type=str, default="", choices=("style", "")
    )
    model_params.add_argument(
      "--learn_default_feature", action="store_true", default=True
    )
    model_params.add_argument(
      "--use_camera", action="store_true", default=False
    )

    model_params.add_argument("--pp_pixel", type=int, default=128,
                              help='K: the number of points to conisder in the z-buffer.'
                              )
    model_params.add_argument("--tau", type=float, default=1.0,
                              help='gamma: the power to raise the distance to.'
                              )
    model_params.add_argument(
      "--use_gt_depth", action="store_true", default=False
    )
    model_params.add_argument(
      "--train_depth", action="store_true", default=False
    )
    model_params.add_argument(
      "--only_high_res", action="store_true", default=False
    )
    model_params.add_argument(
      "--use_inverse_depth", action="store_true", default=False,
      help='If true the depth is sampled as a long tail distribution, else the depth is sampled uniformly. Set to true if the dataset has points that are very far away (e.g. a dataset with landscape images, such as KITTI).'
    )
    model_params.add_argument(
      "--ndf",
      type=int,
      default=64,
      help="# of discrim filters in first conv layer",
    )
    model_params.add_argument(
      "--use_xys", action="store_true", default=False
    )
    model_params.add_argument(
      "--output_nc",
      type=int,
      default=3,
      help="# of output image channels",
    )
    model_params.add_argument("--norm_G", type=str, default="batch")
    model_params.add_argument(
      "--ngf",
      type=int,
      default=64,
      help="# of gen filters in first conv layer",
    )
    model_params.add_argument(
      "--radius",
      type=float,
      default=4,
      help="Radius of points to project",
    )
    model_params.add_argument(
      "--voxel_size", type=int, default=64, help="Size of latent voxels"
    )
    model_params.add_argument(
      "--num_upsampling_layers",
      choices=("normal", "more", "most"),
      default="normal",
      help="If 'more', adds upsampling layer between the two middle resnet blocks. "
           + "If 'most', also add one more upsampling + resnet layer at the end of the generator",
    )

  def add_data_parameters(self):
    dataset_params = self.parser.add_argument_group("data")
    dataset_params.add_argument("--dataset", type=str, default="mp3d")
    dataset_params.add_argument(
      "--use_semantics", action="store_true", default=False
    )
    dataset_params.add_argument(
      "--config",
      type=str,
      default="./data_readers/mp3d.yaml",
    )
    dataset_params.add_argument(
      "--current_episode_train", type=int, default=-1
    )
    dataset_params.add_argument(
      "--current_episode_val", type=int, default=-1
    )
    dataset_params.add_argument("--min_z", type=float, default=0.5)
    dataset_params.add_argument("--max_z", type=float, default=10.0)
    dataset_params.add_argument("--W", type=int, default=256)
    dataset_params.add_argument(
      "--images_before_reset", type=int, default=1000
    )
    dataset_params.add_argument(
      "--image_type",
      type=str,
      default="translation_z",
      choices=(
        "both",
        "translation",
        "rotation",
        "outpaint",
        "fixedRT_baseline",
        "translation_z"
      ),
    )
    dataset_params.add_argument("--max_angle", type=int, default=45)
    dataset_params.add_argument(
      "--use_z", action="store_true", default=False
    )
    dataset_params.add_argument(
      "--use_inv_z", action="store_true", default=False
    )
    dataset_params.add_argument(
      "--use_rgb_features", action="store_true", default=False
    )
    dataset_params.add_argument(
      "--use_alpha", action="store_true", default=False
    )
    dataset_params.add_argument(
      "--normalize_image", action="store_true", default=False
    )
    dataset_params.add_argument("--carla_dist", type=str, default="1")

  def add_train_parameters(self):
    training = self.parser.add_argument_group("training")
    training.add_argument("--num_workers", type=int, default=0)
    training.add_argument("--start-epoch", type=int, default=0)
    training.add_argument("--num-accumulations", type=int, default=1)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--lr_d", type=float, default=1e-3 * 2)
    training.add_argument("--lr_g", type=float, default=1e-3 / 2)
    training.add_argument("--momentum", type=float, default=0.9)
    training.add_argument("--beta1", type=float, default=0)
    training.add_argument("--beta2", type=float, default=0.9)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--init", type=str, default="")

    training.add_argument(
      "--use_multi_hypothesis", action="store_true", default=False
    )
    training.add_argument("--num_hypothesis", type=int, default=1)
    training.add_argument("--z_dim", type=int, default=128)
    training.add_argument(
      "--netD", type=str, default="multiscale", help="(multiscale)"
    )
    training.add_argument(
      "--niter",
      type=int,
      default=100,
      help="# of iter at starting learning rate. This is NOT the total #epochs."
           + " Total #epochs is niter + niter_decay",
    )
    training.add_argument(
      "--niter_decay",
      type=int,
      default=10,
      help="# of iter at starting learning rate. This is NOT the total #epochs."
           + " Totla #epochs is niter + niter_decay",
    )

    training.add_argument(
      "--losses", type=str, nargs="+", default=['1.0_l1', '10.0_content']
    )
    training.add_argument(
      "--discriminator_losses",
      type=str,
      default="pix2pixHD",
      help="(|pix2pixHD|progressive)",
    )
    training.add_argument(
      "--lambda_feat",
      type=float,
      default=10.0,
      help="weight for feature matching loss",
    )
    training.add_argument(
      "--gan_mode", type=str, default="hinge", help="(ls|original|hinge)"
    )

    training.add_argument(
      "--load-old-model", action="store_true", default=False
    )
    training.add_argument(
      "--load-old-depth-model", action="store_true", default=False
    )
    training.add_argument("--old_model", type=str, default="")
    training.add_argument("--old_depth_model", type=str, default="")

    training.add_argument(
      "--no_ganFeat_loss",
      action="store_true",
      help="if specified, do *not* use discriminator feature matching loss",
    )
    training.add_argument(
      "--no_vgg_loss",
      action="store_true",
      help="if specified, do *not* use VGG feature matching loss",
    )
    training.add_argument("--resume", action="store_true", default=False)

    training.add_argument(
      "--log-dir",
      type=str,
      default="./checkpoint/viewsynthesis3d/%s/",
    )

    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--continue_epoch", type=int, default=0)
    training.add_argument("--max_epoch", type=int, default=500)
    training.add_argument("--folder_to_save", type=str, default="outpaint")
    training.add_argument(
      "--model-epoch-path",
      type=str,
      default="/%s/%s/models/lr%0.5f_bs%d_model%s_spl%s/noise%s_bn%s_ref%s_d%s_"
              + "camxys%s/_init%s_data%s_seed%d/_multi%s_losses%s_i%s_%s_vol_gan%s/",
    )
    training.add_argument(
      "--run-dir",
      type=str,
      default="/%s/%s/runs/lr%0.5f_bs%d_model%s_spl%s/noise%s_bn%s_ref%s_d%s_"
              + "camxys%s/_init%s_data%s_seed%d/_multi%s_losses%s_i%s_%s_vol_gan%s/",
    )
    training.add_argument("--suffix", type=str, default="")
    training.add_argument(
      "--render_ids", type=int, nargs="+", default=[0]
    )
    training.add_argument("--gpu_ids", type=str, default="0")

  def parse(self, arg_str=None):
    if arg_str is None:
      args = self.parser.parse_args()
    else:
      args = self.parser.parse_args(arg_str.split())

    arg_groups = {}
    for group in self.parser._action_groups:
      group_dict = {
        a.dest: getattr(args, a.dest, None)
        for a in group._group_actions
      }
      arg_groups[group.title] = group_dict

    return (args, arg_groups)


class HabitatImageGenerator(torch.utils.data.Dataset):

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

    self.worker_id = 0
    self.split = split
    opts = ArgumentParser().parse(arg_str="")[0]
    opts.train_data_path = (
      "./data_readers/scene_episodes/mp3d_train/dataset_one_ep_per_scene.json.gz"
    )
    opts.val_data_path = (
      "./data_readers/scene_episodes/mp3d_val/dataset_one_ep_per_scene.json.gz"
    )
    opts.test_data_path = (
      "./data_readers/scene_episodes/mp3d_test/dataset_one_ep_per_scene.json.gz"
    )
    opts.scenes_dir = "/group/30042/ozhengchen/"#ft_local/data_360/v1/tasks"  # this should store mp3d

    self.opts = opts

    self.num_views = seq_len
    self.vectorize = vectorize

    self.image_generator = None

    # Part of hacky code to have train/val
    self.episodes = None
    self.restarted = True
    self.train = True

    self.rng = np.random.RandomState(seed)
    self.seed = opts.seed

    self.fixed_val_images = [None] * 32  # Keep 32 examples
    self.full_width = full_width
    self.full_height = full_height
    self.depth_to_dist_cache = {}
    self.seq_len = seq_len
    self.reference_idx = reference_idx
    self.m3d_dist = m3d_dist
    self.use_rand = use_rand
    # self.num=99999999
    print("Matterport3D distance:", m3d_dist)

  def __len__(self):
    return 2**31

  def __restart__(self):
    print("-----SPLIT_____" + self.split)
    if self.vectorize:
      self.image_generator = RandomImageGenerator(
        self.split,
        self.opts.render_ids[
          self.worker_id % len(self.opts.render_ids)
          ],
        self.opts,
        vectorize=self.vectorize,
        seed=self.worker_id + self.seed,
        full_width=self.full_width,
        full_height=self.full_height,
        reference_idx=self.reference_idx,
        m3d_dist=self.m3d_dist,
        use_rand=self.use_rand
      )
      self.image_generator.env.reset()
    else:
      self.image_generator = RandomImageGenerator(
        self.split,
        self.opts.render_ids[
          self.worker_id % len(self.opts.render_ids)
          ],
        self.opts,
        vectorize=self.vectorize,
        seed=torch.randint(100, size=(1,)).item(),
        full_width=self.full_width,
        full_height=self.full_height,
        reference_idx=self.reference_idx,
        m3d_dist=self.m3d_dist
      )
      self.image_generator.env.reset()
      self.rng = np.random.RandomState(
        torch.randint(100, size=(1,)).item()
      )

    if not (self.vectorize):
      if self.episodes is None:
        self.rng.shuffle(self.image_generator.env.episodes)
        self.episodes = self.image_generator.env.episodes
      self.image_generator.env.reset()
      self.num_samples = 0

  def restart(self, train):

    if not (self.vectorize):
      if train:
        self.image_generator.env.episodes = self.episodes[
                                            0: int(0.8 * len(self.episodes))
                                            ]
      else:
        self.image_generator.env.episodes = self.episodes[
                                            int(0.8 * len(self.episodes)):
                                            ]

      # randomly choose an environment to start at (as opposed to always 0)
      self.image_generator.env._current_episode_index = self.rng.randint(
        len(self.episodes)
      )
      print(
        "EPISODES A ",
        self.image_generator.env._current_episode_index,
        flush=True,
      )
      self.image_generator.env.reset()
      print(
        "EPISODES B ",
        self.image_generator.env._current_episode_index,
        flush=True,
      )

  def totrain(self, epoch=0):
    self.restarted = True
    self.train = True
    self.seed = epoch

  def toval(self, epoch=0):
    self.restarted = True
    self.train = False
    self.val_index = 0
    self.seed = epoch

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

  def __getitem__(self, item):#index
    if self.cfg["debug"]:
      data=np.load('test_data.npz', allow_pickle=True)
      rgb_panos=data["rgb_panos"]
      depth_panos=data["depth_panos"]
      rots=data["rots"]
      trans=data["trans"]
      self.images  = data["rgb_panos"]#.to(args.device)
      self.depths = data["depth_panos"]#.to(args.device)
      self.rots = data["rots"]#.to(args.device)
      self.trans = data["trans"]#.to(args.device)
    else:
      if not (self.train) and (self.val_index < len(self.fixed_val_images)):
        if self.fixed_val_images[self.val_index]:
          data = self.fixed_val_images[self.val_index]
          self.val_index += 1
          return data

      if self.image_generator is None:
        print(
          "Restarting image_generator.... with seed %d in train mode? %s"
          % (self.seed, self.train),
          flush=True,
        )
        self.__restart__()

      if self.restarted:
        self.restart(self.train)
        self.restarted = False

      # Ignore the item and just generate an image
      data = self.image_generator.get_sample(item, self.num_views, self.train)

      if not (self.train) and (self.val_index < len(self.fixed_val_images)):
        self.fixed_val_images[self.val_index] = data
        self.val_index += 1
      #
      # data["rgb_panos"]
      self.images  = data["rgb_panos"]#.to(args.device)
      self.depths = data["depth_panos"]#.to(args.device)
      self.rots = data["rots"]#.to(args.device)
      self.trans = data["trans"]#.to(args.device)
      
      np.savez('test_data.npz', rgb_panos=data["rgb_panos"], \
            depth_panos=data["depth_panos"], rots=data["rots"], trans=data["trans"])

    # rgb = data["rgb_panos"]
    # os.makedirs("./vis_debug",exist_ok=True)
    # for idx in range(len(rgb)):
    #   tmp_rgb = np.uint8(rgb[idx]*255)
    #   cv2.imwrite("./vis_debug/rgb_"+str(idx)+".jpg", tmp_rgb)    



    ids_all = [0, 2]
    
    # if self.cfg["debug"]:
    src_dict={
      0: 2,
      2: 0,
    }
    if self.split == "train":
      # que_id = 0  # 2
      # if np.random.random()>0.25:
      #   que_id = ids_all[np.random.randint(0,len(ids_all))]
      #   # else:
      #   # que_id = 0
      #   ref_ids_all = list(set([0, 2]) - set([que_id])) # [0, 2]
      #   print("que_id, ref_ids:", que_id, ref_ids_all)
      #   src_ids = [src_dict[ref_id] for ref_id in ref_ids_all]
      # else:
      que_id = 1
      ref_ids_all = [0, 2]#list(set([0, 2]) - set([que_id]))
      src_ids = [src_dict[ref_id] for ref_id in ref_ids_all]
      print("que_id, ref_ids:", que_id, ref_ids_all)        
    else:
      que_id = 1
      ref_ids_all = [0, 2]#list(set([0, 2]) - set([que_id]))
      src_ids = [src_dict[ref_id] for ref_id in ref_ids_all]
    
    database = M3DDatabase(self.cfg, data)
    #self.get_database_ref_que_ids(data, index)
    # ref_ids = self.select_working_views(database, que_id, ref_ids_all)

    # src_imgs_info used in construction of cost volume
    # ref_imgs_info, ref_cv_idx, ref_real_idx = build_src_imgs_info_select(database,ref_ids,ref_id)
    
    #todo: ref_cv_idx, ref_real_idx

    
    if self.cfg["debug"]:
      ref_imgs_info = build_imgs_info(database, ref_ids_all, has_depth=True)#?
      que_imgs_info = build_imgs_info(database, [que_id], has_depth=True)

    else:
      ref_imgs_info = build_imgs_info(database, ref_ids_all)#?
      que_imgs_info = build_imgs_info(database, [que_id])
    
    if self.split=="train":
      # data augmentation
      depth_range_all = np.concatenate([ref_imgs_info['depth_range'],que_imgs_info['depth_range']],0)
      depth_all, mask_all = None, None
      # depth_range_all = self.random_change_depth_range(depth_range_all, depth_all, mask_all, database.database_name)

      # ref_imgs_info['depth_range'] = depth_range_all[:-1]
      # que_imgs_info['depth_range'] = depth_range_all[-1:]
      
      if self.cfg['use_depth_loss_for_all'] and self.cfg['use_depth']:
        if not database.database_name.startswith('gso'):
          ref_imgs_info['true_depth'] = ref_imgs_info['depth']

      # if self.cfg['use_consistent_depth_range']:
      #     self.consistent_depth_range(ref_imgs_info, que_imgs_info)

      # generate coords
      
    if self.split=="train":
      que_mask_cur = que_imgs_info['masks'][0,0]>0 #
      # 1, 512, 2
      coords = get_coords_mask(que_mask_cur, self.cfg['train_ray_num'], self.cfg['foreground_ratio']).reshape([1,-1,2])
    else:
      qn, _, hn, wn = que_imgs_info['imgs'].shape
      coords = np.stack(np.meshgrid(np.arange(wn),np.arange(hn)),-1)
      coords = coords.reshape([1,-1,2]).astype(np.float32)
    que_imgs_info['coords'] = coords

    # ref_imgs_info = pad_imgs_info(ref_imgs_info,self.cfg['ref_pad_interval'])
    

    # don't feed depth to gpu
    # import ipdb;ipdb.set_trace()
    if self.cfg["debug"]:
      pass
    else:
      if not self.cfg['use_depth']:
        if 'depth' in ref_imgs_info: ref_imgs_info.pop('depth')
        if 'depth' in que_imgs_info: que_imgs_info.pop('depth')
        if 'true_depth' in ref_imgs_info: ref_imgs_info.pop('true_depth')

    # src_imgs_info = ref_imgs_info.copy()
    assert len(ref_imgs_info["imgs"]) <= 2, "Only ref_num<=2 is supported now"
    src_imgs_info = build_imgs_info(database, src_ids)#?
    #todo?
    # ref_real_idx = [0, 1]
    # ref_cv_idx = np.asarray([[0, 1],
    #                         [1, 0]])
    # ref_imgs_info = imgs_info_slice(ref_imgs_info, ref_real_idx)
    # ref_imgs_info['nn_ids'] = ref_cv_idx

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