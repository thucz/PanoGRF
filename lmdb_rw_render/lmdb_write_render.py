import lmdb
import cv2
import os
import distro
import numpy as np
# Pytorch Imports
import torch
# from progress.bar import Bar
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# import tqdm
# from data_readers.carla_reader import CarlaReader
import sys
sys.path.append('./')
from data_readers.habitat_data_neuray_ft import HabitatImageGeneratorFT

from helpers import my_torch_helpers
from models import loss_lib
import numpy as np
import random
import argparse
from utils.base_utils import load_cfg
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
        if args["dataset"] == "carla":
            train_data = CarlaReader(
              args["carla_path"],
              width=self.full_width,
              height=self.full_height,
              towns=["Town01", "Town02", "Town03", "Town04"],
              min_dist=args["carla_min_dist"],
              max_dist=args["carla_max_dist"],
              seq_len=2,
              reference_idx=1,
              use_meters_depth=True,
              interpolation_mode=args["interpolation_mode"],
              sampling_method="dense")
          # print("Size of training set: %d" % (len(train_data),))
            train_dataloader = DataLoader(
              train_data,
              batch_size=args["batch_size"],
              shuffle=True,
              num_workers=4)
        elif args["dataset"] == "m3d":
            train_data = HabitatImageGeneratorFT(
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
            
            val_data = HabitatImageGeneratorFT(
              args,
              args["mode"],
              seq_len = args["seq_len"],
              reference_idx = args["reference_idx"],
              full_width=self.full_width,
              full_height=self.full_height,
              m3d_dist=args["m3d_dist"]) #, freeview=args['freeview'], offset=args['offset'])
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
        
        if mode=="train":
          k = args["total_cnt"]/20000 * (args["height"]//256)**2        
          map_size = int(100000000000 * 3/2 * 2 * k)
        elif mode=="val":
          # for 1000 samples
          k = args["total_cnt"]/1000 * (args["height"]//256)**2
          map_size = int(5000000000*3/2 * 2 * k)
        else:
          k = args["total_cnt"]/1000 * (args["height"]//256)**2
          map_size = int(5000000000 *3/2 * 2 * k)
        

        env_path = save_dir+'/lmdb_render_'+mode+"_"+str(args["width"])+"x"+str(args["height"])+"_seq_len_"+str(args["seq_len"])+ \
          "_m3d_dist_"+str(args["m3d_dist"])
        
        if "freeview" in self.config and self.config["freeview"]:
          env_path+="_freeview_"+str(self.config["offset"][0])+","+str(self.config["offset"][1])+","+str(self.config["offset"][2])
        
        self.env_path = env_path
        self.env = lmdb.open(env_path, map_size=map_size, writemap=True)
        print("map_size nbytes:", map_size)
        if mode:
            if mode=="train":
                self.loader=app.train_data_loader
            elif mode=="val":
                self.loader=app.val_data_loader
            else:
                self.loader=app.val_data_loader

    def save_data(self, data, idx):
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
        

  
        self.write_lmdb(base_key, "rgb_panos", panos)
        self.write_lmdb(base_key, "depth_panos", depths)
        self.write_lmdb(base_key, "rots", rots)
        self.write_lmdb(base_key, "trans", trans)        
        self.write_lmdb(base_key, "rgb_cubes", panos_cubes)        
        self.write_lmdb(base_key, "depth_cubes", depths_cubes)
        self.write_lmdb(base_key, "rots_cubes", rots_cubes)
        self.write_lmdb(base_key, "trans_cubes", trans_cubes)
    def write_lmdb(self, base_key, key_post, data):
        key = base_key+","+key_post
        key_byte = key.encode("ascii")
        self.txn.put(key_byte, data)  
                 
    def iter_all(self, total_cnt):
        self.txn = self.env.begin(write = True)        
        for i, data in enumerate(self.loader):
          print("i:", i)
          if i>=total_cnt:
            break;
          self.save_data(data, i)
        self.txn.commit()
        self.env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    flags = parser.parse_args()

    app = App()
    args = app.start(flags)
    os.makedirs(args["save_dir"], exist_ok=True)
    lmdb_inst=WriteLMDB(args, args["mode"], args["save_dir"], app)
    lmdb_inst.iter_all(args["total_cnt"]+1)

