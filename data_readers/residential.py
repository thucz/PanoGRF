import torch
import cv2
import numpy as np
class ResidentialDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.data_list = []
        for idx in range(3):
            self.data_dir="/group/30042/ozhengchen/ft_local/neuray360_data/tmp/residential/ricoh_mini/"+str(idx)+"_perspective_all"
            # pass
            data = torch.load(self.data_dir+"/all.t7")
            train_views = [0, 8] #[3,5]
            val_view = [4]
            all_views = sorted(train_views+val_view)
            
            self.data ={
                "rgbs": data["rgbs"][all_views],
                "cube_rgbs": data["cube_rgbs"][all_views],
                "c2w": data["c2w"][all_views],
                "cube_c2w": data['cube_c2w'][all_views]           
            }
            # rgb->c2w:
            # 0->5
            # 1->1
            # 2->4
            # 3->3
            # 4->2
            # 5->0
            # self.data["cube_c2w"][:, 0] = self.data["cube_c2w"][:, 5]
            # self.data["cube_c2w"][:, 2] = self.data["cube_c2w"][:, 4]
            # def swap(data)
            def swap(arr, idx1, idx2):
                tmp = arr[:, idx1].clone()
                arr[:, idx1] = arr[:, idx2]
                arr[:, idx2] = tmp
                return arr
            def rectify(arr):
                arr = swap(arr, 0, 5)
                arr = swap(arr, 2, 4)                
                return arr
            self.data["cube_c2w"] = rectify(self.data["cube_c2w"])
            self.data_list.append(self.data)
    def __getitem__(self, idx):#index
        # Ignore the item and just generate an image
        data = self.data_list[idx]
        return data
    def __len__(self):
        return len(self.data_list)

