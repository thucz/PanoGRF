import os
import gzip
import math
import copy
import json
import argparse
from typing import Dict
import torchvision
import habitat
import habitat.datasets.pointnav.pointnav_dataset as mp3d_dataset
import numpy as np
import quaternion
import torch
import torchvision.transforms as transforms
import tqdm
from habitat.config.default import get_config
from scipy.spatial.transform.rotation import Rotation
import cv2
from habitat.datasets import make_dataset
import random
import torch.nn.functional as F

from data_readers.mhabitat import vector_env
from helpers import my_helpers
import numpy as np

random.seed(1234)
np.random.seed(1234)
torch.random.manual_seed(1234)

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.obs_transformers import Cube2Equirect

# save_dir="/group/30042/ozhengchen/replica_data"

def save_data_np(data, final_save_dir, sub_dir):
    os.makedirs(final_save_dir+"/"+sub_dir+"/", exist_ok=True)
    # import ipdb;ipdb.set_trace()
    seq_len = data["rgb_panos"].shape[0]
    # cv2.imwrite(final_save_dir+"/"+sub_dir+"/", panos)
    for seq_idx in range(seq_len):
        rgb = data["rgb_panos"][seq_idx]
        depth = data["depth_panos"][seq_idx]
        cv2.imwrite(final_save_dir+"/"+sub_dir+"/rgb_0.jpg", np.uint8(rgb[..., ::-1]*255))
        d_min = depth.min()
        d_max = depth.max()
        d_grey = np.uint8((depth - d_min)/(d_max-d_min)*255)
        d_rgb = cv2.applyColorMap(d_grey, cv2.COLORMAP_JET)
        cv2.imwrite(final_save_dir+"/"+sub_dir+"/depth_0.jpg", d_rgb)
        # pass
    panos = data["rgb_panos"]#.to(args.device)
    depths = data["depth_panos"]#.to(args.device)
    rots = data["rots"]#.to(args.device)
    trans = data["trans"]#.to(args.device)
    rgb_cubes = data["rgb_cubes"]
    depth_cubes = data["depth_cubes"]
    trans_cubes = data["trans_cubes"]
    rots_cubes = data["rots_cubes"]

    np.savez(final_save_dir+"/"+sub_dir+"/data.npz", rgb_panos=panos, depth_panos=depths, rots=rots, trans=trans, rgb_cubes=rgb_cubes, \
                    depth_cubes=depth_cubes, trans_cubes = trans_cubes, rots_cubes=rots_cubes)


class Options:
    def __init__(self, dataset_name):
        self.basedir="/group/30042/ozhengchen"
        self.config = self.basedir+'/replica_make/dataset/pointnav/mp3d.yaml'
        self.num_views = 3
        self.replica_dist = 0.5
        self.episodes_dir = self.basedir+'/synsin/data/scene_episodes' #'/data/teddy/Datasets/synsin_scene_episodes/'
        # wide-baseline
        self.image_type = "translation_z"#"translation"

        # "fixedRT_baseline"
        self.render_ids = [0, 1]
        self.seed = 1
        self.resolution = [512, 1024]
        self.normalize_image = False
        self.use_semantics = False
        self.dataset = dataset_name
        # Spherical data settings
        self.num_sides = 6
        # light field resolution
        self.lf_width = 9
        self.lf_height = 9
        self.lf_baseline = 0.1
        self.orientations = [[0, math.pi, 0],       # back
                             [-0.5*math.pi, 0, 0],   # down
                             [0, 0, 0, ],             # front
                             [0, math.pi / 2, 0],    # right
                             [0, 3/2*math.pi, 0],    # left
                             [0.5*math.pi, 0, 0],    # upward
                             ]
        
        if dataset_name == 'mp3d':
            self.train_data_path = self.episodes_dir+"/mp3d_train/dataset_one_ep_per_scene.json.gz"
            self.test_data_path = self.episodes_dir+"/mp3d_test/dataset_one_ep_per_scene.json.gz"
            self.val_data_path = self.episodes_dir+"/mp3d_val/dataset_one_ep_per_scene.json.gz"
            self.scenes_dir = "/data/teddy/Datasets/matterport_3d/"  # this should store mp3d
        elif dataset_name == "replica":
            self.train_data_path = self.episodes_dir+"/replica_train/dataset_one_ep_per_scene.json.gz"
            #"/data/teddy/Datasets/one_episode_per_scene/replica/train/dataset_one_ep_per_scene.json.gz"
            self.test_data_path = self.episodes_dir+"/replica_test/dataset_one_ep_per_scene.json.gz" #"/data/teddy/Datasets/one_episode_per_scene/replica/test/dataset_one_ep_per_scene.json.gz"
            # self.val_data_path = "/data/teddy/Datasets/one_episode_per_scene/replica/val/dataset_one_ep_per_scene.json.gz"
            self.scenes_dir = '/group/30042/public_datasets/replica/data/'#'/data/teddy/Datasets/replica/'

def make_config(config, gpu_id, split, data_path, pers_resolution, scenes_dir, opts, sensors):
    config = get_config(config)
    config.defrost()
    config.TASK.NAME = "Nav-v0"
    config.TASK.MEASUREMENTS = []
    config.DATASET.SPLIT = split
    # print(dir(config.DATASET))
    # print(dir(config.DATASET.__deprecated_keys__))
    # print((config.DATASET.DEPRECATED_KEYS))
    config.DATASET.DATA_PATH = data_path
    config.DATASET.SCENES_DIR = scenes_dir
    config.HEIGHT = pers_resolution #resolution[0]
    config.WIDTH = pers_resolution #resolution[1]
    # sensor_map = {'RGB': config.SIMULATOR.RGB_SENSOR,
    #               'DEPTH': config.SIMULATOR.DEPTH_SENSOR, }
    # import ipdb;ipdb.set_trace()
    for sensor in sensors:
        config.SIMULATOR[sensor]["HEIGHT"] = pers_resolution
        config.SIMULATOR[sensor]["WIDTH"] = pers_resolution
        config.SIMULATOR[sensor]["POSITION"] = np.array([0, 0, 0])
    config.SIMULATOR.AGENT_0.SENSORS = sensors

    config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False    
    # config.SIMULATOR.RGB_SENSOR.HFOV = 90 #?
    # config.SIMULATOR.RGB_SENSOR.VFOV = 90
    config.SIMULATOR.DEPTH_SENSOR.HFOV = 90
    # config.SIMULATOR.DEPTH_SENSOR.VFOV = 90
    # config.SIMULATOR.RGB_SENSOR.ORIENTATION = opts.orientations[0]
    # config.SIMULATOR.DEPTH_SENSOR.ORIENTATION = opts.orientations[0]
    # print(dir(config.SIMULATOR.DEPTH_SENSOR))
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = pers_resolution #opts.resolution[0]
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = pers_resolution #opts.resolution[1]
    config.SIMULATOR.RGB_SENSOR.HEIGHT = pers_resolution #opts.resolution[0]
    config.SIMULATOR.RGB_SENSOR.WIDTH = pers_resolution #opts.resolution[1]
    # print((config.SIMULATOR.RGB_SENSOR.__dict__.keys()))
    # print((config.SIMULATOR.RGB_SENSOR.__dict__.keys()))
    # exit()
    # sensor_uuids = []
    # for sensor_type, sensor_config in sensor_map.items():
    #     for cam_id in range(opts.num_sides):
    #         uuid = f'{sensor_type.lower()}_key_{cam_id}'
    #         camera_config = copy.deepcopy(sensor_config)
    #         camera_config.ORIENTATION = opts.orientations[cam_id]
    #         camera_config.HEIGHT = pers_resolution #opts.resolution[0]
    #         camera_config.WIDTH = pers_resolution #opts.resolution[1]
    #         camera_config.UUID = uuid.lower()
    #         sensor_uuids.append(camera_config.UUID)
    #         setattr(config.SIMULATOR, uuid, camera_config)
    #         config.SIMULATOR.AGENT_0.SENSORS.append(uuid)

        # obs_trans = baseline_registry.get_obs_transformer("CubeMap2Equirec")
        # cube2equirec = obs_trans(sensor_uuids, (opts.resolution[0], opts.resolution[0]), camera_config["HEIGHT"])
        # setattr(config.SIMULATOR,f'{sensor_type}_{cube2equirec}', cube2equirec)

    # for sensor in sensors:
    #     config.SIMULATOR[sensor]["HEIGHT"] = resolution
    #     config.SIMULATOR[sensor]["WIDTH"] = resolution
    config.TASK.HEIGHT = pers_resolution #resolution[0]
    config.TASK.WIDTH = pers_resolution #resolution[1]
    config.SIMULATOR.TURN_ANGLE = 15
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.1  # in metres
    
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 2 ** 32
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
    return config #, sensor_uuids

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--part', default=0, type=int, help='for train pass [1-4], this is to run 4 different processes, each will run subsets of the total episodes')
arg_parser.add_argument('--split', default='train', type=str,
                        help='train or test')
arg_parser.add_argument('--dist', default=0.5, type=float,
                        help='distance')

args = arg_parser.parse_args()
final_save_dir="/group/30042/ozhengchen/baselines_data/replica_"+str(args.dist)


split = args.split
class ReplicaHabitat():
    def __init__(self):

        opts = Options(dataset_name='replica')
        # opts.lf_width = 3
        # opts.lf_height = 3
        opts.replica_dist = args.dist
        if args.part == 1:
            opts.episodes_list = [0, 1, 2, 3]
        elif args.part == 2:
            opts.episodes_list = [4, 5, 6]
        elif args.part == 3:
            opts.episodes_list = [7, 8, 9]
        elif args.part == 4:
            opts.episodes_list = [10, 11, 12]
        else:#for test
            opts.episodes_list = [0, 1, 2, 3, 4]
        self.train_offset = 0
        if split == 'test':
            self.train_offset = 13 # test
            opts.episodes_list = [i for i in opts.episodes_list if i<5]
        opts.use_semantics = False
        opts.normalize_image = False
        opts.save_depth = True

        # change the resolution here
        # opts.resolution = [1024, 1024]
        # opts.final_resolution = [512, 1024]
        opts.resolution = [512, 1024] #
        opts.final_resolution = [512, 1024] # final Spherical Image resolutoin
        self.height = 512
        self.width = 1024
        self.reference_idx = 1
        width = self.width
        height = self.height
        theta, phi = np.meshgrid((np.arange(width) + 0.5) * (2 * np.pi / width),
                             (np.arange(height) + 0.5) * (np.pi / height))
        uvs, uv_sides = my_helpers.spherical_to_cubemap(theta.reshape(-1),
                                                    phi.reshape(-1))
        # self.width = width
        # self.height = height
        self.uvs = uvs.reshape(self.height, self.width, 2)
        self.uv_sides = uv_sides.reshape(self.height, self.width)
        self.depth_to_dist_cache = {}
        opts.batch_size = 1
        opts.min_depth_threshold = 0.0 #0.90  # if the closest object is close than
        opts.seed = 0
        if split == 'train':
            opts.iters_per_episode = 2
        else:
            opts.iters_per_episode = 2

        # target_path = os.path.join(
        #     f'{save_dir}/dense_lf_1024_512/{opts.dataset}')
        # os.makedirs(target_path, exist_ok=True)

        rng = np.random.RandomState(torch.randint(100, size=(1,)).item())

        # resolution = opts.resolution
        pers_resolution = 256
        self.pers_resolution = pers_resolution
        if split == "train":
            data_path = opts.train_data_path
        elif split == "val":
            data_path = opts.val_data_path
        elif split == "test":
            data_path = opts.test_data_path
        else:
            raise Exception("Invalid split")
        # , sensor_uuids
        sensors = ["RGB_SENSOR", "DEPTH_SENSOR"]

        config = make_config(
            opts.config,
            0,
            split,
            data_path,
            pers_resolution,
            opts.scenes_dir,
            opts,
            sensors
        )
        # obs_trans = baseline_registry.get_obs_transformer('CubeMap2Equirect')#revise
        # equirec_size = (opts.resolution[0], opts.resolution[1])
        # depth_sensor_uuids = [k for k in sensor_uuids if 'depth_key' in k]
        # color_sensor_uuids = [k for k in sensor_uuids if 'rgb_key' in k]

        # color_cube2equirec = copy.deepcopy(
        #     obs_trans(color_sensor_uuids, equirec_size, channels_last=False)).to('cuda:0')
        # depth_cube2equirec = copy.deepcopy(
        #     obs_trans(depth_sensor_uuids, equirec_size, channels_last=False)).to('cuda:0')
        # color_cube2equi = Cube2Equirect(
        #     opts.resolution[0], opts.resolution[1]) #h, w

        data_dir = os.path.join(
            opts.episodes_dir, opts.dataset + "_" + split
            )

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_path = os.path.join(data_dir, "dataset_one_ep_per_scene.json.gz")
        print("One ep per scene", flush=True)

        # if not (os.path.exists(data_path)): #revise
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
            json_ = dataset.to_json().encode("utf-8")
            with gzip.GzipFile(data_path, "w") as fout:
                fout.write(json_)
        print("Finished dataset...", flush=True)
        with gzip.open(data_path, "rt") as f:
            f_str = f.read()
            f_dict = json.loads(f_str)
            # filter unwanted episodes
            new_dict = {}
            new_dict['episodes'] = []
            episodes_ = f_dict['episodes']
            for epi_ in episodes_:
                if epi_['episode_id'] in opts.episodes_list:
                    new_dict['episodes'].append(epi_)
            with gzip.open(f'./temp_{str(opts.episodes_list).replace(", ", "_").replace(" ", "_")}.json.gz', "wt") as f1:
                #  open(f'', 'w') as fp:
                json.dump(new_dict, f1)
            with gzip.open(f'./temp_{str(opts.episodes_list).replace(", ", "_").replace(" ", "_")}.json.gz', "rt") as f2:
                # with open(, 'r') as f2:
                dataset.from_json(f2.read())

            # import ipdb;ipdb.set_trace()
            for i in range(0, len(dataset.episodes)):
                # self.scenes_dir
                dataset.episodes[i].scene_id = dataset.episodes[i].scene_id.replace('/checkpoint/ow045820/data/replica/',
                                            opts.scenes_dir).replace('/checkpoint/erikwijmans/data/mp3d/',
                                                '/data/teddy/Datasets/matterport_3d/mp3d/')

        config.TASK.SENSORS = ["POINTGOAL_SENSOR"]
        config.freeze()

        # Now look at vector environments
        # env = habitat.Env(config=config, dataset=dataset)
        # Now look at vector environments
        # vectorize:
        # configs, datasets = _load_datasets(
        #     (
        #         opts.config,
        #         gpu_id,
        #         split,
        #         data_path,
        #         sensors,
        #         resolution,
        #         opts.scenes_dir,
        #     ),
        #     dataset,
        #     data_path,
        #     opts.scenes_dir + '/mp3d/',
        #     num_workers=num_parallel_envs,
        #     )
        configs = [config]
        datasets = [dataset]
        num_envs = len(configs)
        env_fn_args = tuple(zip(configs, datasets, range(num_envs)))
        env = vector_env.VectorEnv(
            env_fn_args=env_fn_args,
            multiprocessing_start_method="forkserver",
        )
        self.env = env

        self.opts = opts

    # rng.shuffle(env.episodes)
    # env.episodes = env.episodes[9:10]
    # for epi_idx in opts.episodes_list:
    # print('opts.episodes_list .... ', opts.episodes_list)
    # import ipdb;ipdb.set_trace()
    def generate(self):
        opts = self.opts
        save_count = 0
        env_sim = self.env #.sim
        # if isTrain:
        #     index = index % self.num_train_envs
        index = 0 #index for dataloader workers(vector_env)
        env = self.env 
        
        # data_idx = 0
        

        for epi_idx, epi_suffix in enumerate(opts.episodes_list):
            env.reset()
            # episodes = env.episodes
            # import ipdb;ipdb.set_trace()
            env._current_episode_index = epi_idx #[epi_idx].episode_id
            # env._current_episode = episodes[epi_idx]
            # print(dir(env))
            # exit()
            # exit()
            # print(epi_idx, episodes[epi_idx])
            # _current_episode_index
            skipped, total = 0, 0
            # index = 0
            # import ipdb;ipdb.set_trace()
            iter_idx = 0
            if epi_suffix in [2, 4, 5, 7, 8, 10, 12]:
                iters_per_episode = 12
            else:
                iters_per_episode = opts.iters_per_episode
            
            for itr in tqdm.tqdm(range(iters_per_episode), total=iters_per_episode):
                #
                depths = []
                rgbs = []
                translations = []
                rotations = []
                #added
                depth_cubes = []
                rgb_cubes = []
                rots_cubes = []
                trans_cubes = []
                # import ipdb;ipdb.set_trace()                
                orig_location = np.array(env_sim.sample_navigable_point(index))
                rand_angle = 0
                orig_rotation = [0, np.sin(rand_angle / 2), 0, np.cos(rand_angle / 2)]
                # orig_location = np.array(rand_location).copy()
                # orig_rotation = np.array(rand_rotation).copy()

                data = {}
                # device = f'cuda:{0}'
                # for h in range(opts.lf_height):
                #     for w in range(opts.lf_width):

                for i in range(0, opts.num_views): #
                    # position = copy.deepcopy(rand_location)

                    # position[0] = position[0] - 1*w*opts.lf_baseline
                    # position[1] = position[1] - 1*h*opts.lf_baseline
                    # position[2] = position[2]
                    rand_location = orig_location.copy()
                    rand_rotation = orig_rotation.copy()
                    
                    if opts.image_type == "translation_z":
                        movement_deltas = {
                            0: opts.replica_dist,
                            1: 0.0,
                            2: -opts.replica_dist
                        }
                        rand_location[[2]] = (
                            orig_location[[2]] + movement_deltas[i]
                        )
                    else:
                        raise ValueError("Unknown image type")
                    
                    # obs = env_sim.get_observations_at(
                    #     position=rand_location,
                    #     rotation=rand_rotation,
                    #     keep_agent_at_new_pose=False,
                    # )
                    # import ipdb;ipdb.set_trace()
                    cubemap_rotations = [
                        Rotation.from_euler('x', 90, degrees=True),  # Top
                        Rotation.from_euler('y', 0, degrees=True), #front
                        Rotation.from_euler('y', -90, degrees=True), # left
                        Rotation.from_euler('y', -180, degrees=True), #back
                        Rotation.from_euler('y', -270, degrees=True), #right
                        Rotation.from_euler('x', -90, degrees=True)  # Bottom
                    ]

                    rgb_cubemap_sides = []
                    depth_cubemap_sides = []
                    rotations_cubemap_sides=[]#q_vec
                    locations_cubemap_sides=[]#t_vec
                    rand_location = rand_location + np.array([0, 1.25, 0]) #
                    rand_rotation = Rotation.from_quat(rand_rotation) #
                    for j in range(6):
                        my_rotation = (rand_rotation * cubemap_rotations[j]).as_quat()
                        obs = env.get_observations_at(
                            index,
                            position=rand_location,
                            rotation=my_rotation.tolist()
                        )
                        # import ipdb;ipdb.set_trace()
                        normalized_rgb = obs["rgb"].astype(np.float32) / 255.0
                        rgb_cubemap_sides.append(normalized_rgb)
                        depth_cubemap_sides.append(obs["depth"])
                        locations_cubemap_sides.append(rand_location)
                        rotations_cubemap_sides.append((Rotation.from_quat(my_rotation)).as_matrix())
                    locations_cubemap_sides = np.stack(locations_cubemap_sides, axis=0)
                    rotations_cubemap_sides = np.stack(rotations_cubemap_sides, axis=0)

                    rgb_cubemap_sides = np.stack(rgb_cubemap_sides, axis=0)
                    rgb_erp_image = self.stitch_cubemap(rgb_cubemap_sides, clip=True)
                    # import ipdb;ipdb.set_trace()
                    depth_cubemap_sides = np.stack(depth_cubemap_sides, axis=0)
                    
                    depth_erp_image = self.stitch_cubemap(depth_cubemap_sides, clip=False, depth_input=True)


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
                rotation_offset = Rotation.from_euler('x', np.pi, degrees=False).as_matrix()
                for i in range(translations.shape[0]):
                    for j in range(6):
                        trans_cubes[i, j] = np.linalg.inv(rotations[reference_idx]) @ (
                        trans_cubes[i,j] - translations[reference_idx]
                        )
                    
                # Rotate cameras 180 degrees
                rotation_offset = Rotation.from_euler('x', np.pi, degrees=False).as_matrix()
                rots_cubes[ :, 1:5] = np.einsum("...ij,jk->...ik", rots_cubes[:, 1:5], rotation_offset)

                # points_cam_other = rots_other@(points_w[:3, :]-trans_other)
                # import ipdb;ipdb.set_trace()
                trans_cubes = trans_cubes[..., np.newaxis]
                trans_cubes = np.einsum("...ij,...jk->...ik", rots_cubes, -trans_cubes)
                # rots_cubes[i,j] = rotations[reference_idx] @ np.linalg.inv(rots_cubes[i, j])#
                # rots_other@(points_w[:3, :]-trans_other)
                    
                for i in range(translations.shape[0]):
                    if i != reference_idx:
                        translations[i] = -np.linalg.inv(rotations[reference_idx]) @ (
                            translations[i] - translations[reference_idx])
                        rotations[i] = rotations[reference_idx] @ np.linalg.inv(rotations[i])#
                translations[reference_idx] = 0.0 * translations[reference_idx]
                rotations[reference_idx] = np.eye(3)
                
                # self.num_samples += 1

                rgbs = np.stack(rgbs, axis=0)
                depths = np.stack(depths, axis=0)
                # import ipdb;ipdb.set_trace()
                depths = depths[..., 0:1]
                # depths = self.new_zdepth_to_dist(depths)

                # depths = self.zdepth_to_distance(depths)

                #revised
                # import ipdb;ipdb.set_trace()
                # cv2.interpolate()
                #panos
                new_cube_width = rgbs.shape[1]//2 #256 for pano 128 for cube
                new_rgb_cubes = np.zeros((depth_cubes.shape[0], depth_cubes.shape[1], new_cube_width, new_cube_width, 3))
                # depth_cubes = depth_cubes.squeeze(-1)
                new_depth_cubes = np.zeros((depth_cubes.shape[0], depth_cubes.shape[1], new_cube_width, new_cube_width, 1))

                for seq_idx in range(depth_cubes.shape[0]):
                    for cube_idx in range(depth_cubes.shape[1]):
                        # import ipdb;ipdb.set_trace()
                        new_depth_cubes[seq_idx, cube_idx] = cv2.resize(depth_cubes[seq_idx, cube_idx], (new_cube_width, new_cube_width), cv2.INTER_LINEAR)[..., np.newaxis]
                        new_rgb_cubes[seq_idx, cube_idx] = cv2.resize(rgb_cubes[seq_idx, cube_idx], (new_cube_width, new_cube_width), cv2.INTER_LINEAR)
                # import ipdb;ipdb.set_trace()
                data = {
                    "rgb_panos": rgbs[:, :, :, :3],
                    "rots": rotations,
                    "trans": translations,
                    "depth_panos": depths[:, :, :, 0],
                    "rgb_cubes": new_rgb_cubes.astype(np.float32),
                    "depth_cubes": new_depth_cubes.astype(np.float32),
                    "rots_cubes": rots_cubes,
                    "trans_cubes": trans_cubes,

                }                
                panos = data["rgb_panos"]#.to(args.device)
                depths = data["depth_panos"]#.to(args.device)
                rots = data["rots"]#.to(args.device)
                trans = data["trans"]#.to(args.device)
                rgb_cubes = data["rgb_cubes"]
                depth_cubes = data["depth_cubes"]
                trans_cubes = data["trans_cubes"]
                rots_cubes = data["rots_cubes"]
                # np.savez("./test_data.npz", panos=panos, depths=depths, rots=rots, trans=trans, rgb_cubes=rgb_cubes, \
                #     depth_cubes=depth_cubes, trans_cubes = trans_cubes, rots_cubes=rots_cubes)  
                sub_dir=str(epi_suffix+self.train_offset)+"_"+str(iter_idx)
                # sub_dir=str(epi_suffix)+"_"+str(iter_idx)
                                
                np.savez("./replica_test_data.npz", panos=panos, depths=depths, rots=rots, trans=trans, rgb_cubes=rgb_cubes, \
                    depth_cubes=depth_cubes, trans_cubes = trans_cubes, rots_cubes=rots_cubes)
                
                select_mode=True
                if select_mode:
                    if epi_suffix in [2,4, 5, 7, 8, 10, 12]:
                        subset={(2, 7), (4, 0), (5, 1), (7, 7), (8, 5), (10, 5), (12, 6)}
                        if (epi_suffix, iter_idx) in subset:
                            sub_dir=str(epi_suffix+self.train_offset)+"_"+str(0)
                            save_data_np(data, final_save_dir, sub_dir)
                    else:
                        if iter_idx == 0:
                            save_data_np(data, final_save_dir, sub_dir)
                else:
                    save_data_np(data, final_save_dir, sub_dir)
                iter_idx += 1
                # break;

                # import ipdb;ipdb.set_trace()
                # seq_len = panos.shape


                # depth_map = data[f'depths_{mid_h}_{mid_w}']
                # depth_map = depth_map.view(-1)
                # min_depth = torch.min(depth_map[depth_map.gt(0)])
                # total += 1
                # # import ipdb;ipdb.set_trace()
                # print("min_depth:", min_depth)
                # print("opts.min_depth_threshold:", opts.min_depth_threshold)
                # if min_depth > opts.min_depth_threshold:
                #     for h in range(opts.lf_height):
                #         for w in range(opts.lf_width):
                #             color_img = data[f'rgbs_{h}_{w}']
                #             depth_map = data[f'depths_{h}_{w}'] if opts.save_depth else None
                #             if not opts.resolution == opts.final_resolution:
                #                 color_img = F.interpolate(
                #                     color_img, size=opts.final_resolution, mode='bilinear')
                #                 if not depth_map is None:
                #                     depth_map = F.interpolate(
                #                         depth_map, size=opts.final_resolution, mode='bilinear')
                #             folder = os.path.join(
                #                 target_path, f'episode_{str(epi_suffix+train_offset).zfill(5)}', str(itr).zfill(5))
                #             os.makedirs(folder, exist_ok=True)
                #             save_sample(folder, itr, h,
                #                         w, color_img[0], depth=depth_map[0])
                # else:
                #     skipped += 1
                #     print(f'Skipping {skipped} out of {total}')
                #     save_count += 1
                # if save_count % 1000 == 0:
                #     print(f'{save_count} samples out of {opts.iters_per_episode*len(env.episodes)} saved')
                # index += 1

    def stitch_cubemap(self, cubemap, clip=True, depth_input=False):
        """Stitches a single cubemap into an equirectangular image.
        Args:
        cubemap: Cubemap images as 6xHxWx3 arrays.
        clip: Clip values to [0, 1].
        Returns:
        Single equirectangular image as HxWx3 image.
        """
        cube_height, cube_width = cubemap.shape[1:3]
        if depth_input:
            # import ipdb;ipdb.set_trace()
            #cubemap:
            # 6, 256, 256, 1
            # new_cubemap = np.zeros(cubemap
            # for i in range(cubemap.shape[0]):
            cubemap = self.new_zdepth_to_dist(cubemap)
            # import ipdb;ipdb.set_trace()

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
                (np.arange(width) + 0.5) * (2 * np.pi / width), (np.arange(height) + 0.5) * (np.pi / height))
                
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
    def new_zdepth_to_dist(self, zdepths):
            # if True:
        # correct depth: from z-coord to ray length
        height = self.pers_resolution #opts.resolution[0]
        width = self.pers_resolution #opts.resolution[1]    
        # import ipdb;ipdb.set_trace()    
        y_2d = torch.linspace(
            0.5, height-0.5, height).view(1, height, 1, 1)
        x_2d = torch.linspace(
            0.5, width-0.5, width).view(1, 1, width, 1)
        x_2d = x_2d.expand(1, height, width, 1)
        y_2d = y_2d.expand(1, height, width, 1)
        ones_ = torch.ones_like(x_2d)
        hom_2d = torch.cat([x_2d, y_2d, ones_],
                            dim=3).unsqueeze(4) #.to('cuda:0')

        fx = 0.5*width/math.tan(math.pi/4)
        fy = 0.5*height/math.tan(math.pi/4)
        cy, cx = (height-1)/2, (width-1)/2
    
        k_matrix = torch.FloatTensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        k_matrix = k_matrix.view(1, 3, 3)
        k_inv = torch.inverse(k_matrix).view(
            1, 1, 1, 3, 3)#.to('cuda:0')
        
        # for key_, depth_map in depth_obs_dict.items():
        depth_maps = torch.from_numpy(zdepths)

        # import ipdb;ipdb.set_trace()
        new_depths = torch.zeros_like(depth_maps)
        for idx in range(depth_maps.shape[0]):
            depth_map = depth_maps[idx]
            h_2d = hom_2d.clone()
            rays = torch.matmul(k_inv, h_2d).squeeze(-1)
            z = depth_map.view(1, height, width, 1)
            scaled_rays = z*rays
            ray_len = torch.norm(
                scaled_rays, p=2, dim=-1, keepdim=True)
            new_depths[idx:idx+1] = ray_len.view(
                1, height, width, 1)
        # import ipdb;ipdb.set_trace()
        return new_depths.numpy()


repli = ReplicaHabitat()
repli.generate()