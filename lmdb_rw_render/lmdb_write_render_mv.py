import lmdb
import cv2
import os
import distro
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('./')
from data_readers.habitat_data_neuray_ft_mv import HabitatImageGeneratorFTMultiView

from helpers import my_torch_helpers
from models import loss_lib
import numpy as np
import random
import argparse
from utils.base_utils import load_cfg
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class App:
    """Main app class"""
    default_cfg={
        # "batch_size": 1,
        # "width": 512,
        # "height": 256,
        # "dataset": "m3d",
        # "carla_min_dist": 2, 
        # "carla_max_dist": 100,
        # "min_depth": 0.1,
        # "max_depth": 10,        
        # "m3d_dist": 1.0,        
        # "num_workers": 0,
        # "seq_len": 2,
        # "reference_idx": 1,
        # "total_cnt": 200000,
        # "aug": False,
    }
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.checkpoint_manager = None
        self.args = None
        self.writer = None

        # Attributes to hold training data.
        self.train_data = None
        self.train_data_loader = None

        # Attributes to hold validation data.
        self.val_data = None

    def start(self, flags):
        """Starts the training."""
        cfg=load_cfg(flags.cfg)
        self.cfg={**self.default_cfg,**cfg}
        args = self.cfg
        self.args = args
        self.full_width = args["width"]
        self.full_height = args["height"]
        seed = 2022
        setup_seed(seed)

        self.load_training_data()
        self.load_validation_data()
        return args

    def load_training_data(self):
        """Loads training data."""
        args = self.args
        # Prepare dataset loaders for train and validation datasets.
        # if args["dataset"] == "carla":
        #     train_data = CarlaReader(
        #       args["carla_path"],
        #       width=self.full_width,
        #       height=self.full_height,
        #       towns=["Town01", "Town02", "Town03", "Town04"],
        #       min_dist=args["carla_min_dist"],
        #       max_dist=args["carla_max_dist"],
        #       seq_len=2,
        #       reference_idx=1,
        #       use_meters_depth=True,
        #       interpolation_mode=args["interpolation_mode"],
        #       sampling_method="dense")
        #   # print("Size of training set: %d" % (len(train_data),))
        #     train_dataloader = DataLoader(
        #       train_data,
        #       batch_size=args["batch_size"],
        #       shuffle=True,
        #       num_workers=4)
        if args["dataset"] == "m3d":
            train_data = HabitatImageGeneratorFTMultiView(
              args,
              split="train",
              seq_len = args["seq_len"],
              reference_idx = args["reference_idx"],
              full_width=self.full_width,
              full_height=self.full_height,
              m3d_dist=args["m3d_dist"],
              aug=args["aug"]
            )
            train_dataloader = DataLoader(
              dataset=train_data,
              num_workers=args["num_workers"],
              batch_size=args["batch_size"],
              shuffle=False,
              drop_last=True,
              pin_memory=True,
            )
        train_data.cache_depth_to_dist(args["height"], args["width"])
        self.train_data = train_data
        self.train_data_loader = train_dataloader
    def load_validation_data(self):
        """Loads validation data."""
        args = self.args
        if args["dataset"] == "carla":
            towns = ["Town05"]
            if args["script_mode"] == "eval_depth_test":
              towns = ["Town06"]
            val_data = CarlaReader(
              args["carla_path"],
              width=self.full_width,
              height=self.full_height,
              towns=towns,
              min_dist=args["carla_min_dist"],
              max_dist=args["carla_max_dist"],
              seq_len=2,
              reference_idx=1,
              use_meters_depth=True,
              interpolation_mode=args["interpolation_mode"])
        elif args["dataset"] == "m3d":
            # if args["script_mode"] == "eval_depth_test":
            #     val_data = HabitatImageGeneratorFT(
            #       args,
            #       "val",
            #       seq_len = args["seq_len"],
            #       reference_idx = args["reference_idx"],
            #       full_width=self.full_width,
            #       full_height=self.full_height,
            #       m3d_dist=args["m3d_dist"])
            # else:
            val_data = HabitatImageGeneratorFTMultiView(
              args,
              args["mode"],
              seq_len = args["seq_len"],
              reference_idx = args["reference_idx"],
              full_width=self.full_width,
              full_height=self.full_height,
              m3d_dist=args["m3d_dist"])
        self.val_data = val_data
        self.val_data_loader = DataLoader(
              dataset=val_data,
              num_workers=0,
              batch_size=1,
              shuffle=False,
            )

class WriteLMDB():
    def __init__(self, args, mode, save_dir, app):
        self.config = args
        # self.if_train = if_train
        # 200000/100=2000
        # 200000/1000 = 200
        
        if mode=="train":
          # for 200000 samples          
          # for 20000 samples
          # 16*300G=4.8T
          # 4*300G=1.2T #
          # k = args["total_cnt"]/20000 * (args["height"]//256)**2          
          # # import ipdb;ipdb.set_trace()
          # map_size = int(100000000000 * 3/2 * 2 * k * args['seq_len']/3*1.5) #1000000000000 * 3/2   #5000000000#1000000000000 #5000000000 #1000000000000 #/ 2000 = 500000000
          k = args["total_cnt"]/1000 * (args["height"]//256)**2
          map_size = int(5000000000*3/2 * 2 * k * args['seq_len']/3*1.5) #5000000000 #1000000000000 / 200 = 

        elif mode=="val":
          # for 1000 samples
          k = args["total_cnt"]/1000 * (args["height"]//256)**2
          map_size = int(5000000000*3/2 * 2 * k * args['seq_len']/3) #5000000000 #1000000000000 / 200 = 
        else:
          k = args["total_cnt"]/1000 * (args["height"]//256)**2
          map_size = int(5000000000 *3/2 * 2 * k * args['seq_len']/3) #1000000000000 / 200 =
        # (H*W*map_cnt+pts_cnt*4*2)*3*2*len(self.data_lst)#bytes#每个数字按照2bytes(float16, uint16)来算；以3倍存放
        # Byte
        # 2*256*512*3*4 = 3145728
        # 2*256*512*1*4 + = 4194304
        # 2*3*3 =  18
        # 2*3 = 6
        # (4194304+24)*200000 = 838, 865, 600, 000

        env_path = save_dir+'/lmdb_render_'+mode+"_"+str(args["width"])+"x"+str(args["height"])+"_seq_len_"+str(args["seq_len"])+ \
          "_m3d_dist_"+str(args["m3d_dist"])+"_mv_"+str(args["reference_idx"])
        
        self.env_path = env_path
        # import ipdb;ipdb.set_trace()
        self.env = lmdb.open(env_path, map_size=map_size, writemap=True)#1T=1099511627776
        print("map_size nbytes:", map_size)
        
        if mode:
            if mode=="train":
                self.loader=app.train_data_loader
            elif mode=="val":
                self.loader=app.val_data_loader
            else:
                self.loader=app.val_data_loader

    def save_data(self, data, idx):
        # import ipdb;ipdb.set_trace()
        panos = data["rgb_panos"]#.to(args["device"])
        depths = data["depth_panos"]#.to(args["device"])
        rots = data["rots"]#.to(args["device"])
        trans = data["trans"]#.to(args["device"])
        panos_cubes = data["rgb_cubes"]#.to(args["device"])
        depths_cubes = data["depth_cubes"]#.to(args["device"])
        rots_cubes = data["rots_cubes"]#.to(args["device"])
        trans_cubes = data["trans_cubes"]#.to(args["device"])
        
        base_key = str(idx)
        
        panos = panos[0].data.cpu().numpy()
        depths = depths[0].data.cpu().numpy()
        rots = rots[0].data.cpu().numpy()
        trans = trans[0].data.cpu().numpy()        
        panos_cubes = panos_cubes[0].data.cpu().numpy()
        depths_cubes = depths_cubes[0].data.cpu().numpy()
        rots_cubes = rots_cubes[0].data.cpu().numpy()
        trans_cubes = trans_cubes[0].data.cpu().numpy()

        # imgs_steps = np.ascontiguousarray(imgs)
        # import ipdb;ipdb.set_trace()
        self.write_lmdb(base_key, "rgb_panos", panos)
        self.write_lmdb(base_key, "depth_panos", depths)
        self.write_lmdb(base_key, "rots", rots)
        self.write_lmdb(base_key, "trans", trans)        

        # import ipdb;ipdb.set_trace()
        self.write_lmdb(base_key, "rgb_cubes", panos_cubes)
        self.write_lmdb(base_key, "depth_cubes", depths_cubes)
        self.write_lmdb(base_key, "rots_cubes", rots_cubes)
        self.write_lmdb(base_key, "trans_cubes", trans_cubes)        

    def write_lmdb(self, base_key, key_post, data):
        key = base_key+","+key_post
        key_byte = key.encode("ascii")
        self.txn.put(key_byte, data)  
                 
    # def __getitem__(self, idx):  
    #     video_name, frame_idx, video_len = self.data_lst[idx]
    #     data = self.read_data(self.data_lst[idx], idx)
    #     return data 
    # def __len__(self):
    #     return len(self.data_lst)
    def iter_all(self, total_cnt):
        self.txn = self.env.begin(write = True)
        pbar=tqdm(total=total_cnt,bar_format='{r_bar}')

        for i, data in enumerate(self.loader):
          # import ipdb;ipdb.set_trace()
          print("i:", i)
          if i>=total_cnt:
            break;
          # import ipdb;ipdb.set_trace()
          self.save_data(data, i)
          pbar.update(1)

          # import ipdb;ipdb.set_trace()

        # import ipdb;ipdb.set_trace()
        pbar.close()

        self.txn.commit()
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)#'configs/train/gen/neuray_gen_depth_train.yaml')
    flags = parser.parse_args()

    app = App()
    args = app.start(flags)
    # save_dir="/group/30042/ozhengchen/lmdb"
    os.makedirs(args["save_dir"], exist_ok=True)
    lmdb_inst=WriteLMDB(args, args["mode"], args["save_dir"], app)
    # total_cnt=200000
    lmdb_inst.iter_all(args["total_cnt"]+1)

