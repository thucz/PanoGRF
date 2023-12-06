import argparse
import os
import lpips

import torch
from skimage.io import imread
from tqdm import tqdm
import numpy as np

from utils.base_utils import color_map_forward
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from network.metrics import WSPSNR
class Evaluator:
    def __init__(self):
        self.loss_fn_alex = lpips.LPIPS(net='vgg').cuda().eval()
        # self.loss_fn_alex = lpips.LPIPS(net='alex').cuda().eval()
        self.wspsnr_calculator = WSPSNR()

    def eval_metrics_img(self,gt_img, pr_img):
        gt_img = color_map_forward(gt_img)
        pr_img = color_map_forward(pr_img)
        psnr = tf.image.psnr(tf.convert_to_tensor(gt_img), tf.convert_to_tensor(pr_img), 1.0, )
        ssim = tf.image.ssim(tf.convert_to_tensor(gt_img), tf.convert_to_tensor(pr_img), 1.0, )
        with torch.no_grad():
            gt_img_th = torch.from_numpy(gt_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1
            pr_img_th = torch.from_numpy(pr_img).cuda().permute(2,0,1).unsqueeze(0) * 2 - 1            
            score = float(self.loss_fn_alex(gt_img_th, pr_img_th).flatten()[0].cpu().numpy())
        y_pred = torch.from_numpy(pr_img).unsqueeze(0)  # RGB
        y_true = torch.from_numpy(gt_img).unsqueeze(0)  
        # ws_psnr(self, y_pred, y_true, max_val=1.0)
        # import ipdb;ipdb.set_trace()
        ws_psnr = self.wspsnr_calculator.ws_psnr(
            y_pred, y_true, max_val=1.0)  # input: B, H, W, C
        
        return ws_psnr.item(), float(psnr), float(ssim), score


    def eval(self, flags):
        results=[]
        scene_num = flags.scene_num
        # "/group/30042/ozhengchen/NeuRay-spherical-broken-ae-erp+tp/data/render/m3d/"
        
        for scene_idx in range(scene_num):
            dir_gt = flags.dir_prefix+"-"+str(scene_idx)+"-"+"gt"
            dir_pr = flags.dir_prefix+"-"+str(scene_idx)
            num = len(os.listdir(dir_gt))        
            for k in tqdm(range(0, num)):
                pr_img = imread(f'{dir_pr}/{k}-nr_fine.jpg')
                gt_img = imread(f'{dir_gt}/{k}.jpg')
                ws_psnr, psnr, ssim, lpips_score = self.eval_metrics_img(gt_img, pr_img)
                results.append([ws_psnr,psnr,ssim,lpips_score])
        ws_psnr, psnr, ssim, lpips_score = np.mean(np.asarray(results),0)
        msg=f'ws_psnr {ws_psnr:.4f} psnr {psnr:.4f} ssim {ssim:.4f} lpips {lpips_score:.4f}'
        print(msg)
        with open(flags.database_name+"_cubes_metric.txt", "w") as fp:
            fp.write(msg)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dir_gt', type=str, default='data/render/fern/gt')
    # parser.add_argument('--dir_pr', type=str, default='data/render/fern/neuray_gen_depth-pretrain-eval')
    parser.add_argument('--scene_num', type=int, default=10)
    parser.add_argument('--dir_prefix', type=str, default='')    
    parser.add_argument('--database_name', type=str, default='m3d')    

    flags = parser.parse_args()

    evaluator = Evaluator()
    evaluator.eval(flags)
