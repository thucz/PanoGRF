# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Taken from https://github.com/facebookresearch/splitnet

import gzip
import os
from typing import Dict

import habitat
import habitat.datasets.pointnav.pointnav_dataset as mp3d_dataset
import numpy as np
import quaternion
import torch
import torchvision.transforms as transforms
import tqdm
from habitat.config.default import get_config
from habitat.datasets import make_dataset
from scipy.spatial.transform.rotation import Rotation

from data_readers.mhabitat import vector_env
from geometry.camera_transformations import get_camera_matrices
from helpers import my_helpers
from mutils.jitter import jitter_quaternions


def _load_datasets(config_keys, dataset, data_path, scenes_path, num_workers):
  # For each scene, create a new dataset which is added with the config
  # to the vector environment.

  print(len(dataset.episodes))
  datasets = []
  configs = []

  num_episodes_per_worker = len(dataset.episodes) / float(num_workers)

  for i in range(0, min(len(dataset.episodes), num_workers)):
    config = make_config(*config_keys)
    config.defrost()

    dataset_new = mp3d_dataset.PointNavDatasetV1()
    with gzip.open(data_path, "rt") as f:
      dataset_new.from_json(f.read())
      dataset_new.episodes = dataset_new.episodes[
                             int(i * num_episodes_per_worker): int(
                               (i + 1) * num_episodes_per_worker
                             )
                             ]

      for episode_id in range(0, len(dataset_new.episodes)):
        dataset_new.episodes[episode_id].scene_id = \
          dataset_new.episodes[episode_id].scene_id.replace(
            '/checkpoint/erikwijmans/data/mp3d/',
            scenes_path)

    config.SIMULATOR.SCENE = str(dataset_new.episodes[0].scene_id)
    config.freeze()

    datasets += [dataset_new]
    configs += [config]
  return configs, datasets


def make_config(
    config, gpu_id, split, data_path, sensors, resolution, scenes_dir
):
  config = get_config(config)
  config.defrost()
  config.TASK.NAME = "Nav-v0"
  config.TASK.MEASUREMENTS = []
  config.DATASET.SPLIT = split
  # config.DATASET.POINTNAVV1.DATA_PATH = data_path
  config.DATASET.DATA_PATH = data_path
  config.DATASET.SCENES_DIR = scenes_dir
  config.HEIGHT = resolution
  config.WIDTH = resolution
  for sensor in sensors:
    config.SIMULATOR[sensor]["HEIGHT"] = resolution
    config.SIMULATOR[sensor]["WIDTH"] = resolution
    config.SIMULATOR[sensor]["POSITION"] = np.array([0, 0, 0])

  config.TASK.HEIGHT = resolution
  config.TASK.WIDTH = resolution
  config.SIMULATOR.TURN_ANGLE = 15
  config.SIMULATOR.FORWARD_STEP_SIZE = 0.1  # in metres
  config.SIMULATOR.AGENT_0.SENSORS = sensors
  config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False

  config.SIMULATOR.DEPTH_SENSOR.HFOV = 90

  config.ENVIRONMENT.MAX_EPISODE_STEPS = 2 ** 32
  config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
  return config


class RandomImageGenerator(object):
  def __init__(self, split, gpu_id, opts, vectorize=False, seed=0,
               full_width=512, full_height=256,
               reference_idx=1, m3d_dist=1.0,
               use_rand=True) -> None:
    self.vectorize = vectorize

    print("gpu_id", gpu_id)
    resolution = opts.W
    if opts.use_semantics:
      sensors = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    else:
      sensors = ["RGB_SENSOR", "DEPTH_SENSOR"]
    if split == "train":
      data_path = opts.train_data_path
    elif split == "val":
      data_path = opts.val_data_path
    elif split == "test":
      data_path = opts.test_data_path
    else:
      raise Exception("Invalid split")
    
    unique_dataset_name = opts.dataset

    self.num_parallel_envs = 2
    self.use_rand = use_rand

    self.images_before_reset = opts.images_before_reset
    config = make_config(
      opts.config,
      gpu_id,
      split,
      data_path,
      sensors,
      resolution,
      opts.scenes_dir,
    )
    data_dir = os.path.join(
      "data/scene_episodes/", unique_dataset_name + "_" + split
    )
    self.dataset_name = config.DATASET.TYPE
    print(data_dir)
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
    data_path = os.path.join(data_dir, "dataset_one_ep_per_scene.json.gz")
    # Creates a dataset where each episode is a random spawn point in each scene.
    print("One ep per scene", flush=True)
    if not (os.path.exists(data_path)):
      print("Creating dataset...", flush=True)
      dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
      # Get one episode per scene in dataset
      scene_episodes = {}
      for episode in tqdm.tqdm(dataset.episodes):
        if episode.scene_id not in scene_episodes:
          scene_episodes[episode.scene_id] = episode

      scene_episodes = list(scene_episodes.values())
      dataset.episodes = scene_episodes
      if not os.path.exists(data_path):
        # Multiproc do check again before write.
        json = dataset.to_json().encode("utf-8")
        with gzip.GzipFile(data_path, "w") as fout:
          fout.write(json)
      print("Finished dataset...", flush=True)

    # Load in data and update the location to the proper location (else
    # get a weird, uninformative, error -- Affine2Dtransform())
    dataset = mp3d_dataset.PointNavDatasetV1()
    with gzip.open(data_path, "rt") as f:
      dataset.from_json(f.read())

      for i in range(0, len(dataset.episodes)):
        dataset.episodes[i].scene_id = dataset.episodes[i].scene_id.replace(
          '/checkpoint/erikwijmans/data/mp3d/',
          opts.scenes_dir + '/mp3d/')

    config.TASK.SENSORS = ["POINTGOAL_SENSOR"]

    config.freeze()

    self.rng = np.random.RandomState(seed)
    self.reference_idx = reference_idx

    # Now look at vector environments
    if self.vectorize:
      configs, datasets = _load_datasets(
        (
          opts.config,
          gpu_id,
          split,
          data_path,
          sensors,
          resolution,
          opts.scenes_dir,
        ),
        dataset,
        data_path,
        opts.scenes_dir + '/mp3d/',
        num_workers=self.num_parallel_envs,
      )
      num_envs = len(configs)
      env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
      envs = vector_env.VectorEnv(
        env_fn_args=env_fn_args,
        multiprocessing_start_method="forkserver",
      )

      self.env = envs
      self.num_train_envs = int(0.9 * (self.num_parallel_envs))
      self.num_val_envs = self.num_parallel_envs - self.num_train_envs
    else:
      self.env = habitat.Env(config=config, dataset=dataset)
      self.env_sim = self.env.sim
      self.rng.shuffle(self.env.episodes)
      self.env_sim = self.env.sim

    self.num_samples = 0

    # Set up intrinsic parameters
    self.hfov = config.SIMULATOR.DEPTH_SENSOR.HFOV * np.pi / 180.0
    self.W = resolution
    self.K = np.array(
      [
        [1.0 / np.tan(self.hfov / 2.0), 0.0, 0.0, 0.0],
        [0, 1.0 / np.tan(self.hfov / 2.0), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ],
      dtype=np.float32,
    )

    self.invK = np.linalg.inv(self.K)

    self.config = config
    self.opts = opts

    if self.opts.normalize_image:
      self.transform = transforms.Compose(
        [
          transforms.ToTensor(),
          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
      )  # Using same normalization as BigGan
    else:
      self.transform = transforms.ToTensor()

    width = full_width
    height = full_height
    theta, phi = np.meshgrid((np.arange(width) + 0.5) * (2 * np.pi / width),
                             (np.arange(height) + 0.5) * (np.pi / height))
    uvs, uv_sides = my_helpers.spherical_to_cubemap(theta.reshape(-1),
                                                    phi.reshape(-1))
    self.width = width
    self.height = height
    self.uvs = uvs.reshape(height, width, 2)
    self.uv_sides = uv_sides.reshape(height, width)
    self.depth_to_dist_cache = {}
    self.m3d_dist = m3d_dist

  def get_vector_sample(self, index, num_views, isTrain=True):
    if self.num_samples % self.images_before_reset == 0:
      self.env.reset()

    # Randomly choose an index of given environments
    if isTrain:
      index = index % self.num_train_envs
    else:
      index = (index % self.num_val_envs) + self.num_train_envs

    depths = []
    rgbs = []

    orig_location = np.array(self.env.sample_navigable_point(index))
    rand_angle = 0
    if not self.use_rand:
      rand_angle = self.rng.uniform(0, 2 * np.pi)

    orig_rotation = [0, np.sin(rand_angle / 2), 0, np.cos(rand_angle / 2)]#[0, 0, 0, 1]
    # obs = self.env.get_observations_at(
    #   index, position=orig_location, rotation=orig_rotation
    # )
    translations = []
    rotations = []
    #added
    depth_cubes = []
    rgb_cubes = []
    rots_cubes = []
    trans_cubes = []

    for i in range(0, num_views):
      rand_location = orig_location.copy()
      rand_rotation = orig_rotation.copy()
      if self.opts.image_type == "translation_z":
        movement_deltas = {
          0: self.m3d_dist,
          1: 0.0,
          2: -self.m3d_dist
        }
        rand_location[[2]] = (
            orig_location[[2]] + movement_deltas[i]
        )
      else:
        raise ValueError("Unknown image type")

      cubemap_rotations = [
        Rotation.from_euler('x', 90, degrees=True),  # Top
        Rotation.from_euler('y', 0, degrees=True),
        Rotation.from_euler('y', -90, degrees=True),
        Rotation.from_euler('y', -180, degrees=True),
        Rotation.from_euler('y', -270, degrees=True),
        Rotation.from_euler('x', -90, degrees=True)  # Bottom
      ]

      rgb_cubemap_sides = []
      depth_cubemap_sides = []
      rotations_cubemap_sides=[]#q_vec, c2w
      locations_cubemap_sides=[]#t_vec, c2w

      rand_location = rand_location + np.array([0, 1.25, 0]) #
      rand_rotation = Rotation.from_quat(rand_rotation) #
      for j in range(6):
        my_rotation = (rand_rotation * cubemap_rotations[j]).as_quat()
        obs = self.env.get_observations_at(
          index,
          position=rand_location,
          rotation=my_rotation.tolist()
        )
        normalized_rgb = obs["rgb"].astype(np.float32) / 255.0
        rgb_cubemap_sides.append(normalized_rgb)
        depth_cubemap_sides.append(obs["depth"])
        locations_cubemap_sides.append(rand_location)
        rotations_cubemap_sides.append((Rotation.from_quat(my_rotation)).as_matrix())

      locations_cubemap_sides = np.stack(locations_cubemap_sides, axis=0)
      rotations_cubemap_sides = np.stack(rotations_cubemap_sides, axis=0)      

      rgb_cubemap_sides = np.stack(rgb_cubemap_sides, axis=0)
      rgb_erp_image = self.stitch_cubemap(rgb_cubemap_sides, clip=True)
      depth_cubemap_sides = np.stack(depth_cubemap_sides, axis=0)


      depth_erp_image = self.stitch_cubemap(depth_cubemap_sides, clip=False)
      depths += [depth_erp_image]
      rgbs += [rgb_erp_image]

      rotations.append(rand_rotation.as_matrix())#
      translations.append(rand_location)#

      depth_cubes.append(depth_cubemap_sides)
      rgb_cubes.append(rgb_cubemap_sides)
      trans_cubes.append(locations_cubemap_sides)
      rots_cubes.append(rotations_cubemap_sides)
    
    trans_cubes = np.stack(trans_cubes, axis=0).astype(np.float32)
    rots_cubes = np.stack(rots_cubes, axis=0).astype(np.float32)
    depth_cubes = np.stack(depth_cubes, axis=0)#.astype(np.float32)
    rgb_cubes = np.stack(rgb_cubes, axis=0)#.astype(np.float32)

    
    translations = np.stack(translations, axis=0).astype(np.float32)
    rotations = np.stack(rotations, axis=0).astype(np.float32)

    reference_idx = self.reference_idx
    
    #reference:
    # translations[reference_idx], rotations[reference_idx]


    for i in range(translations.shape[0]):#seq_len
      for j in range(6):
        trans_cubes[i, j] = np.linalg.inv(rotations[reference_idx]) @ (
          trans_cubes[i,j] - translations[reference_idx]
        )#c2w
        rots_cubes[i,j] = rotations[reference_idx] @ np.linalg.inv(rots_cubes[i, j])#
    # import ipdb;ipdb.set_trace()

    for i in range(translations.shape[0]):
      if i != reference_idx:
        translations[i] = np.linalg.inv(rotations[reference_idx]) @ (
            translations[i] - translations[reference_idx])#c2w
        rotations[i] = rotations[reference_idx] @ np.linalg.inv(rotations[i])#

    translations[reference_idx] = 0.0 * translations[reference_idx]
    rotations[reference_idx] = np.eye(3)
    cubemap_rotations = [
      Rotation.from_euler('x', 90, degrees=True),  # Top
      Rotation.from_euler('y', 0, degrees=True),
      Rotation.from_euler('y', -90, degrees=True),
      Rotation.from_euler('y', -180, degrees=True),
      Rotation.from_euler('y', -270, degrees=True),
      Rotation.from_euler('x', -90, degrees=True)  # Bottom
    ]
    rots_cubes = []
    trans_cubes = []
    for view_idx in range(num_views):
      rots_cubemap_sides=[]
      trans_cubemap_sides=[]
      for cube_idx in range(6):
        # import ipdb;ipdb.set_trace()
        rots_cubemap_sides.append((Rotation.from_matrix(rotations[view_idx])*cubemap_rotations[cube_idx]).as_matrix())        
        trans_cubemap_sides.append(translations[view_idx])
      rots_cubemap_sides = np.stack(rots_cubemap_sides, axis=0)
      trans_cubemap_sides = np.stack(trans_cubemap_sides, axis=0)
      rots_cubes.append(rots_cubemap_sides)
      trans_cubes.append(trans_cubemap_sides)
    rots_cubes = np.stack(rots_cubes, axis=0)
    trans_cubes = np.stack(trans_cubes, axis=0)
    
    #cubes求相对pose
    self.num_samples += 1

    rgbs = np.stack(rgbs, axis=0)
    depths = np.stack(depths, axis=0)
    depths = depths[..., 0:1]
    depths = self.zdepth_to_distance(depths)
    #revised
    sample = {
      "rgb_panos": rgbs[:, :, :, :3],
      "rots": rotations,
      "trans": translations,
      "depth_panos": depths[:, :, :, 0],
      "rgb_cubes": rgb_cubes,
      "depth_cubes": depth_cubes,
      "rots_cubes": rots_cubes,
      "trans_cubes": trans_cubes,
    }
    return sample

  def get_singleenv_sample(self, num_views) -> Dict[str, np.ndarray]:

    if self.num_samples % self.images_before_reset == 0:
      old_env = self.env._current_episode_index
      self.env.reset()
      print(
        "RESETTING %d to %d \n"
        % (old_env, self.env._current_episode_index),
        flush=True,
      )

    depths = []
    rgbs = []
    cameras = []
    semantics = []

    rand_location = self.env_sim.sample_navigable_point()
    if self.opts.image_type == "fixedRT_baseline":
      rand_angle = self.angle_rng.uniform(0, 2 * np.pi)
    else:
      rand_angle = self.rng.uniform(0, 2 * np.pi)
    rand_rotation = [0, np.sin(rand_angle / 2), 0, np.cos(rand_angle / 2)]
    obs = self.env_sim.get_observations_at(
      position=rand_location,
      rotation=rand_rotation,
      keep_agent_at_new_pose=True,
    )

    for i in range(0, num_views):
      position = rand_location.copy()
      rotation = rand_rotation.copy()
      if self.opts.image_type == "translation":
        position[0] = position[0] + self.rng.rand() * 0.2 - 0.1
      elif self.opts.image_type == "outpaint":
        rotation = quaternion.as_float_array(
          jitter_quaternions(
            quaternion.from_float_array(rand_rotation),
            self.rng,
            angle=10,
          )
        ).tolist()
      elif self.opts.image_type == "fixedRT_baseline":
        rand_location = self.rand_location
        rotation = self.rand_rotation

      else:
        position[0] = position[0] + self.rng.rand() * 0.3 - 0.15
        rotation = quaternion.as_float_array(
          jitter_quaternions(
            quaternion.from_float_array(rand_rotation),
            self.rng,
            angle=10,
          )
        ).tolist()

      obs = self.env_sim.get_observations_at(
        position=position,
        rotation=rotation,
        keep_agent_at_new_pose=True,
      )

      depths += [torch.Tensor(obs["depth"][..., 0]).unsqueeze(0)]
      rgbs += [self.transform(obs["rgb"].astype(np.float32) / 255.0)]

      if "semantic" in obs.keys():
        instance_semantic = torch.Tensor(
          obs["semantic"].astype(np.int32)
        ).unsqueeze(0)
        class_semantic = torch.zeros(instance_semantic.size()).long()

        id_to_label = {
          int(obj.id.split("_")[-1]): obj.category.index()
          for obj in self.env.sim.semantic_annotations().objects
        }

        for id_scene in np.unique(instance_semantic.numpy()):
          class_semantic[instance_semantic == id_scene] = id_to_label[
            id_scene
          ]

        semantics += [class_semantic]

      agent_state = self.env_sim.get_agent_state().sensor_states["depth"]
      rotation = quaternion.as_rotation_matrix(agent_state.rotation)
      position = agent_state.position
      P, Pinv = get_camera_matrices(position=position, rotation=rotation)
      cameras += [{"P": P, "Pinv": Pinv, "K": self.K, "Kinv": self.invK}]

    self.num_samples += 1
    if len(semantics) > 0:
      return {
        "images": rgbs,
        "depths": depths,
        "cameras": cameras,
        "semantics": semantics,
      }

    return {"images": rgbs, "depths": depths, "cameras": cameras}

  def get_sample(self, index, num_views, isTrain):
    if self.vectorize:
      return self.get_vector_sample(index, num_views, isTrain)
    else:
      return self.get_singleenv_sample(num_views)

  def stitch_cubemap(self, cubemap, clip=True):
    """Stitches a single cubemap into an equirectangular image.
    Args:
      cubemap: Cubemap images as 6xHxWx3 arrays.
      clip: Clip values to [0, 1].
    Returns:
      Single equirectangular image as HxWx3 image.
    """
    cube_height, cube_width = cubemap.shape[1:3]

    uvs = self.uvs
    uv_sides = self.uv_sides
    height = self.height
    width = self.width

    skybox_uvs = np.stack(
      (uvs[:, :, 0] * (cube_width - 1), uvs[:, :, 1] * (cube_height - 1)),
      axis=-1)
    final_image = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(0, 6):
      # Grabs a transformed side of the cubemap.
      my_side_indices = np.equal(uv_sides, i)
      final_image[my_side_indices] = my_helpers.bilinear_interpolate(
        cubemap[i, :, :, :], skybox_uvs[my_side_indices, 0],
        skybox_uvs[my_side_indices, 1])
    if clip:
      final_image = np.clip(final_image, 0, 1)
    return final_image

  def zdepth_to_distance(self, depth_image):
    """Converts a depth (z-depth) image to a euclidean distance image.

    Args:
      depth_image: Equirectangular depth image as BxHxWx1 array.

    Returns: Equirectangular distance image.

    """
    batch_size, height, width, channels = depth_image.shape
    cache_key = "_".join((str(height), str(width)))
    self.cache_depth_to_dist(height, width)
    ratio = self.depth_to_dist_cache[cache_key]
    new_depth_image = depth_image * ratio[np.newaxis, :, :, np.newaxis]
    return new_depth_image

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
      # print("focal_len:", focal_len)
      # print("width_center:", width_center)
      # print("height_center:", height_center)
      # import ipdb;ipdb.set_trace()

      diag_dist = np.sqrt((uv_int[:, :, 0] - width_center) ** 2 +
                          (uv_int[:, :,
                           1] - height_center) ** 2 + focal_len ** 2)
      self.depth_to_dist_cache[cache_key] = diag_dist / focal_len
