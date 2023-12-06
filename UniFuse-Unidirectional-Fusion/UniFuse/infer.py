from __future__ import absolute_import, division, print_function
import os
import argparse
import tqdm
import numpy as np
import torch
import cv2
from networks import UniFuse, Equi
import datasets
# from metrics import Evaluator
# from saver import Saver
from preprocess import to_torch_data
parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Test")
parser.add_argument("--image_path", default="", type=str, help="path to the dataset.")
parser.add_argument("--dataset", default="matterport3d", choices=["3d60", "panosuncg", "stanford2d3d", "matterport3d"],
                    type=str, help="dataset to evaluate on.")
parser.add_argument("--load_weights_dir", type=str, help="folder of model to load")
parser.add_argument("--num_workers", type=int, default=4, help="number of dataloader workers")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")

settings = parser.parse_args()
def main():
    max_depth_meters = 10.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_weights_folder = os.path.expanduser(settings.load_weights_dir)
    model_path = os.path.join(load_weights_folder, "model.pth")
    model_dict = torch.load(model_path)
    # # data
    # datasets_dict = {"3d60": datasets.ThreeD60,             
    #                  "panosuncg": datasets.PanoSunCG,       
    #                  "stanford2d3d": datasets.Stanford2D3D, 
    #                  "matterport3d": datasets.Matterport3D} 
    # dataset = datasets_dict[settings.dataset]
    # fpath = os.path.join(os.path.dirname(__file__), "datasets", "{}_{}.txt")
    # test_file_list = fpath.format(settings.dataset, "test")
    # test_dataset = dataset(settings.data_path, test_file_list,
    #                        model_dict['height'], model_dict['width'], is_training=False)
    # test_loader = DataLoader(test_dataset, settings.batch_size, False,
    #                          num_workers=settings.num_workers, pin_memory=True, drop_last=False)
    # num_test_samples = len(test_dataset)
    # num_steps = num_test_samples // settings.batch_size
    # print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")

    # network
    Net_dict = {"UniFuse": UniFuse,
                "Equi": Equi}
    Net = Net_dict[model_dict['net']]

    model = Net(model_dict['layers'], model_dict['height'], model_dict['width'],
                max_depth=max_depth_meters, fusion_type=model_dict['fusion'],
                se_in_fusion=model_dict['se_in_fusion'])
    model.to(device)
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()
    with torch.no_grad():
        inputs = to_torch_data(settings.image_path)
        equi_inputs = inputs["normalized_rgb"].to(device)
        cube_inputs = inputs["normalized_cube_rgb"].to(device)
        outputs = model(equi_inputs, cube_inputs)
        pred_depth = outputs["pred_depth"].detach().cpu()
        # import ipdb;ipdb.set_trace()

        pred_depth = pred_depth.permute((0, 2, 3, 1))[0]
        d_min = torch.min(pred_depth)
        d_max = torch.max(pred_depth)
        d_norm = (pred_depth - d_min) / (d_max - d_min)
        d_norm = np.uint8(d_norm.data.cpu().numpy()*255)
        import ipdb;ipdb.set_trace()

        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        cv2.imwrite("depth_inference.jpg", d_color)

if __name__ == "__main__":
    main()