import torch
import cv2
import numpy as np
class ReplicaWideDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        m3d_dist=cfg["m3d_dist"]
        self.cfg=cfg

        self.data_dir="/group/30042/ozhengchen/baselines_data/replica_"+str(m3d_dist)
        self.scenes_list = list(range(18))
        self.img_wh = [cfg["width"], cfg["height"]]

        self.data_list = self.read_meta()


    def read_meta(self):
        sub_idx=0
        data_list = []
        for scene_idx in self.scenes_list:
            data_path = self.data_dir+"/"+str(scene_idx)+"_"+str(sub_idx)+"/data.npz"
            data = np.load(data_path)
            # import ipdb;ipdb.set_trace() #resize
            panos = data["rgb_panos"]
            depths = data["depth_panos"]
            seq_len = panos.shape[0]
            new_panos = []
            new_depths = []
            # import ipdb;ipdb.set_trace()
            for idx in range(seq_len):
                new_panos.append(cv2.resize(panos[idx], (self.img_wh[0], self.img_wh[1]), cv2.INTER_LINEAR))
                new_depths.append(cv2.resize(depths[idx], (self.img_wh[0], self.img_wh[1]), cv2.INTER_LINEAR))
            
            
            depths = np.array(new_depths)
            panos = np.array(new_panos )
            new_data = {
                "rgb_panos": panos,
                "depth_panos": depths,
                "rots": data["rots"],
                "trans": data["trans"]     
            }
            if "render_cubes" in self.cfg and self.cfg["render_cubes"]:
                # import ipdb;ipdb.set_trace()
                height = data["rgb_cubes"].shape[2]
                assert height == self.cfg['height']//2, "height of cubemap must be 256!"
                new_data.update({
                    "rgb_cubes":data["rgb_cubes"],
                    "depth_cubes":data["depth_cubes"],
                    "trans_cubes": data["trans_cubes"],
                    "rots_cubes": data["rots_cubes"]})

            data_list.append(new_data)
        return data_list

    def __getitem__(self, idx):#index
        
        # Ignore the item and just generate an image
        data = self.data_list[idx]
        return data
    def __len__(self):
        return len(self.data_list)