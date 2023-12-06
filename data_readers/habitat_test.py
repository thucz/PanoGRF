from torch.utils.data import DataLoader
import sys

sys.path.append("./data_readers")
from habitat_data_neuray_ft_aug import HabitatImageGeneratorFT

def load_data(mode, full_width, full_height, m3d_dist, seq_len=3, reference_idx=1):
    # args.dataset_name == "m3d":
    train_data = HabitatImageGeneratorFT(
        args,
        split=mode,
        full_width=full_width,
        full_height=full_height,
        m3d_dist=m3d_dist,
        seq_len=seq_len,
        reference_idx=reference_idx,
    )

    train_dataloader = DataLoader(
        dataset=train_data,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    return train_data, train_dataloader

# mode='train'
mode='test'
train_data, train_loader = load_data(mode, 512, 256, 0.5, seq_len=3, reference_idx=1)


def vis_data():
    loader = train_loader
    data_idx = 0



    # data = train_data.__getitem__(data_idx)
    for i, data in enumerate(loader):
        print("i:", i)
        if i>= data_idx:
            panos = data["rgb_panos"]#.to(args.device)
            depths = data["depth_panos"]#.to(args.device)
            rots = data["rots"]#.to(args.device)
            trans = data["trans"]#.to(args.device)
            rgb_cubes = data["rgb_cubes"]
            depth_cubes = data["depth_cubes"]
            trans_cubes = data["trans_cubes"]
            rots_cubes = data["rots_cubes"]

            import numpy as np
            np.savez("./test_data.npz", panos=panos, depths=depths, rots=rots, trans=trans, rgb_cubes=rgb_cubes, \
                depth_cubes=depth_cubes, trans_cubes = trans_cubes, rots_cubes=rots_cubes)
            import cv2

            # import ipdb;ipdb.set_trace()

            out = np.zeros_like(panos[0, 0])
            cv2.normalize(panos[0, 0].data.cpu().numpy()*255, out, 0, 255, cv2.NORM_MINMAX)

            rgb_out = np.array(out,dtype='uint8')
            cv2.imwrite("pano_np_"+str(data_idx)+".jpg", rgb_out)

            break;

vis_data()
 