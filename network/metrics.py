from pathlib import Path

import torch
from skimage.io import imsave

from network.loss import Loss
from utils.base_utils import color_map_backward, make_dir
from skimage.metrics import structural_similarity
import numpy as np

from utils.draw_utils import concat_images_list
import numpy as np
import torch
import torch.nn.functional as F
import os

def compute_psnr(img_gt, img_pr, use_vis_scores=False, vis_scores=None, vis_scores_thresh=1.5):
    if use_vis_scores:
        mask = vis_scores >= vis_scores_thresh
        mask = mask.flatten()
        img_gt = img_gt.reshape([-1, 3]).astype(np.float32)[mask]
        img_pr = img_pr.reshape([-1, 3]).astype(np.float32)[mask]
        mse = np.mean((img_gt - img_pr) ** 2, 0)

    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = np.mean((img_gt - img_pr) ** 2, 0)
    mse = np.mean(mse)
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr


# class PSNR_SSIM(Loss):
#     default_cfg = {
#         'eval_margin_ratio': 1.0,
#     }
#     def __init__(self, cfg):
#         super().__init__([])
#         self.cfg={**self.default_cfg,**cfg}

#     def __call__(self, data_pr, data_gt, step, **kwargs):
#         rgbs_gt = data_pr['pixel_colors_gt'] # 1,rn,3
#         rgbs_pr = data_pr['pixel_colors_nr'] # 1,rn,3
#         # import ipdb;ipdb.set_trace()

#         if 'que_imgs_info' in data_gt:
#             h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
#         else:
#             h, w = data_pr['que_imgs_info']['imgs'].shape[2:]
#         rgbs_pr = rgbs_pr.reshape([h,w,3]).detach().cpu().numpy()

#         rgbs_pr=color_map_backward(rgbs_pr)

#         rgbs_gt = rgbs_gt.reshape([h,w,3]).detach().cpu().numpy()
#         rgbs_gt = color_map_backward(rgbs_gt)

#         h, w, _ = rgbs_gt.shape
#         h_margin = int(h * (1 - self.cfg['eval_margin_ratio'])) // 2
#         w_margin = int(w * (1 - self.cfg['eval_margin_ratio'])) // 2
#         rgbs_gt = rgbs_gt[h_margin:h - h_margin, w_margin:w - w_margin]
#         rgbs_pr = rgbs_pr[h_margin:h - h_margin, w_margin:w - w_margin]

#         psnr = compute_psnr(rgbs_gt,rgbs_pr)
#         ssim = structural_similarity(rgbs_gt,rgbs_pr,win_size=11,multichannel=True,data_range=255)
#         outputs={
#             'psnr_nr': torch.tensor([psnr],dtype=torch.float32),
#             'ssim_nr': torch.tensor([ssim],dtype=torch.float32),
#         }

#         def compute_psnr_prefix(suffix):
#             if f'pixel_colors_{suffix}' in data_pr:
#                 rgbs_other = data_pr[f'pixel_colors_{suffix}'] # 1,rn,3
#                 # h, w = data_pr['shape']
#                 rgbs_other = rgbs_other.reshape([h,w,3]).detach().cpu().numpy()
#                 rgbs_other=color_map_backward(rgbs_other)
#                 psnr = compute_psnr(rgbs_gt,rgbs_other)
#                 ssim = structural_similarity(rgbs_gt,rgbs_other,win_size=11,multichannel=True,data_range=255)
#                 outputs[f'psnr_{suffix}']=torch.tensor([psnr], dtype=torch.float32)
#                 outputs[f'ssim_{suffix}']=torch.tensor([ssim], dtype=torch.float32)

#         # compute_psnr_prefix('nr')
#         compute_psnr_prefix('dr')
#         compute_psnr_prefix('nr_fine')
#         compute_psnr_prefix('dr_fine')
#         return outputs

# class VisualizeImage(Loss):
#     def __init__(self, cfg):
#         super().__init__([])

#     def __call__(self, data_pr, data_gt, step, **kwargs):
#         if 'que_imgs_info' in data_gt:
#             h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
#         else:
#             h, w = data_pr['que_imgs_info']['imgs'].shape[2:]
#         def get_img(key):
#             rgbs = data_pr[key] # 1,rn,3
#             rgbs = rgbs.reshape([h,w,3]).detach().cpu().numpy()
#             rgbs = color_map_backward(rgbs)
#             return rgbs

#         outputs={}
#         imgs=[get_img('pixel_colors_gt'), get_img('pixel_colors_nr')]
#         # import ipdb; ipdb.set_trace()
#         if 'pixel_colors_dr' in data_pr: imgs.append(get_img('pixel_colors_dr'))
#         if 'pixel_colors_nr_fine' in data_pr: imgs.append(get_img('pixel_colors_nr_fine'))
#         if 'pixel_colors_dr_fine' in data_pr: imgs.append(get_img('pixel_colors_dr_fine'))
#         data_index=kwargs['data_index']
#         model_name=kwargs['model_name']
#         Path(f'data/vis_val/{model_name}').mkdir(exist_ok=True, parents=True)
#         if h<=64 and w<=64:
#             imsave(f'data/vis_val/{model_name}/step-{step}-index-{data_index}.png',concat_images_list(*imgs))
#         else:
#             imsave(f'data/vis_val/{model_name}/step-{step}-index-{data_index}.jpg', concat_images_list(*imgs))
#         # import ipdb;ipdb.set_trace()
#         return outputs

class WSPSNR:
    """Weighted to spherical PSNR"""

    def __init__(self):
        self.weight_cache = {}

    def get_weights(self, height=1080, width=1920):
        """Gets cached weights.
        Args:
            height: Height.
            width: Width.
        Returns:
          Weights as H, W tensor.
        """
        key = str(height) + ";" + str(width)
        if key not in self.weight_cache:
            v = (np.arange(0, height) + 0.5) * (np.pi / height)
            v = np.sin(v).reshape(height, 1)
            v = np.broadcast_to(v, (height, width))
            self.weight_cache[key] = v.copy()
        return self.weight_cache[key]

    def calculate_wsmse(self, reconstructed, reference):
        """
        Calculates weighted mse for a single channel.
        Args:
            reconstructed: Image as B, H, W, C tensor.
            reference: Image as B, H, W, C tensor.
        Returns:
            wsmse
        """
        batch_size, height, width, channels = reconstructed.shape
        weights = torch.tensor(self.get_weights(height, width),
                               device=reconstructed.device,
                               dtype=reconstructed.dtype)
        weights = weights.view(1, height, width, 1).expand(
            batch_size, -1, -1, channels)
        squared_error = torch.pow((reconstructed - reference), 2.0)
        wmse = torch.sum(weights * squared_error, dim=(1, 2, 3)) / torch.sum(
            weights, dim=(1, 2, 3))
        return wmse

    def ws_psnr(self, y_pred, y_true, max_val=1.0):
        """
        Args:Weighted to spherical PSNR.
          y_pred: First image as B, H, W, C tensor.
          y_true: Second image.
          max: Maximum value.
        Returns:Tensor.
        """
        wmse = self.calculate_wsmse(y_pred, y_true)
        ws_psnr = 10 * torch.log10(max_val * max_val / wmse)
        return ws_psnr


class PSNR_SSIM(Loss):
    default_cfg = {
        'eval_margin_ratio': 1.0,
    }

    def __init__(self, cfg):
        super().__init__([])
        self.cfg = {**self.default_cfg, **cfg}
        self.wspsnr_calculator = WSPSNR()

    def __call__(self, data_pr, data_gt, step, **kwargs):
        # import ipdb;ipdb.set_trace()
        if "diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"] and "one_mlp" in self.cfg and self.cfg["one_mlp"]:
            rgbs_gt = data_pr['pixel_colors_gt_fine']  # 1,rn,3
            rgbs_pr = data_pr['pixel_colors_nr_fine']  # 1,rn,3
            # import ipdb;ipdb.set_trace()
            h, w = self.cfg["height"], self.cfg['width']
            # if 'que_imgs_info' in data_gt:
            #     h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
            # else:
            #     h, w = data_pr['que_imgs_info']['imgs'].shape[2:]
        else:
            rgbs_gt = data_pr['pixel_colors_gt']  # 1,rn,3
            rgbs_pr = data_pr['pixel_colors_nr']  # 1,rn,3
            # import ipdb;ipdb.set_trace()
            if 'que_imgs_info' in data_gt:
                h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
            else:
                h, w = data_pr['que_imgs_info']['imgs'].shape[2:]

        rgbs_pr = rgbs_pr.reshape([h, w, 3]).detach().cpu().numpy()
        rgbs_pr = color_map_backward(rgbs_pr)
        rgbs_gt = rgbs_gt.reshape([h, w, 3]).detach().cpu().numpy()
        rgbs_gt = color_map_backward(rgbs_gt)

        h, w, _ = rgbs_gt.shape
        h_margin = 0  # int(h * (1 - self.cfg['eval_margin_ratio'])) // 2
        w_margin = 0  # int(w * (1 - self.cfg['eval_margin_ratio'])) // 2
        rgbs_gt = rgbs_gt[h_margin:h - h_margin, w_margin:w - w_margin]
        rgbs_pr = rgbs_pr[h_margin:h - h_margin, w_margin:w - w_margin]

        psnr = compute_psnr(rgbs_gt, rgbs_pr)
        ssim = structural_similarity(
            rgbs_gt, rgbs_pr, win_size=11, multichannel=True, data_range=255)

        # import ipdb;ipdb.set_trace()
        y_pred = torch.from_numpy(rgbs_pr/255).unsqueeze(0)  # RGB
        y_true = torch.from_numpy(rgbs_gt/255).unsqueeze(0)
        # ws_psnr(self, y_pred, y_true, max_val=1.0)
        ws_psnr = self.wspsnr_calculator.ws_psnr(
            y_pred, y_true, max_val=1.0)  # input: B, H, W, C
        # print("psnr:", psnr)
        # print("ws_psnr:", ws_psnr)
        # print("torch.tensor([psnr],dtype=torch.float32):", torch.tensor([psnr],dtype=torch.float32))
        outputs = {
            'psnr_nr': torch.tensor([psnr], dtype=torch.float32),
            'ssim_nr': torch.tensor([ssim], dtype=torch.float32),
            'wspsnr_nr':  torch.tensor([ws_psnr[0]], dtype=torch.float32)
        }

        def compute_psnr_prefix(suffix):
            if f'pixel_colors_{suffix}' in data_pr:
                rgbs_other = data_pr[f'pixel_colors_{suffix}']  # 1,rn,3
                # h, w = data_pr['shape']
                rgbs_other = rgbs_other.reshape(
                    [h, w, 3]).detach().cpu().numpy()
                rgbs_other = color_map_backward(rgbs_other)
                psnr = compute_psnr(rgbs_gt, rgbs_other)
                
                ssim = structural_similarity(
                    rgbs_gt, rgbs_other, win_size=11, multichannel=True, data_range=255)
                y_pred = torch.from_numpy(rgbs_other/255).unsqueeze(0)  # RGB
                # y_true = torch.from_numpy(rgbs_gt/255).unsqueeze(0)
                # ws_psnr(self, y_pred, y_true, max_val=1.0)
                ws_psnr = self.wspsnr_calculator.ws_psnr(
                    y_pred, y_true, max_val=1.0)  # input: B, H, W, C

                outputs[f'psnr_{suffix}'] = torch.tensor(
                    [psnr], dtype=torch.float32)
                outputs[f'ssim_{suffix}'] = torch.tensor(
                    [ssim], dtype=torch.float32)
                outputs[f'wspsnr_{suffix}'] = torch.tensor(
                    [ws_psnr[0]], dtype=torch.float32)

        # compute_psnr_prefix('nr')

        if "diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"] and "one_mlp" in self.cfg and self.cfg["one_mlp"]:
            outputs["psnr_nr_fine"] = outputs["psnr_nr"]  # one MLP
            # pass
        else:
            compute_psnr_prefix('dr')
            compute_psnr_prefix('nr_fine')
            compute_psnr_prefix('dr_fine')


        def get_img(key):
            rgbs = data_pr[key]  # 1,rn,3
            rgbs = rgbs.reshape([h, w, 3]).detach().cpu().numpy()
            rgbs = color_map_backward(rgbs)
            return rgbs

        # data_index = kwargs['data_index']
        # model_name = kwargs['model_name']
        # # save images for final equi evaluation
        # Path(
        #     f'data/model/{model_name}/equi_evaluation').mkdir(exist_ok=True, parents=True)
        # imsave(f'data/model/{model_name}/equi_evaluation/step-{step}-index-{data_index}-pred.png',
        #        get_img('pixel_colors_nr_fine'))
        # imsave(f'data/model/{model_name}/equi_evaluation/step-{step}-index-{data_index}-gt.png',
        #        get_img('pixel_colors_gt'))
 
        return outputs


class VisualizeImage(Loss):
    def __init__(self, cfg):
        super().__init__([])
        self.cfg = cfg

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if "diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"]:
            h, w = self.cfg["height"], self.cfg["width"]
        else:
            if 'que_imgs_info' in data_gt:
                h, w = data_gt['que_imgs_info']['imgs'].shape[2:]
            else:
                h, w = data_pr['que_imgs_info']['imgs'].shape[2:]

        def get_img(key):
            rgbs = data_pr[key]  # 1,rn,3
            rgbs = rgbs.reshape([h, w, 3]).detach().cpu().numpy()
            rgbs = color_map_backward(rgbs)
            return rgbs

        outputs = {}
        if "diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"]:
            gt_rgb = get_img('pixel_colors_gt_fine')  # h, w, 3
        else:
            gt_rgb = get_img('pixel_colors_gt')  # h, w, 3
        h, w, _ = gt_rgb.shape
        gap = np.zeros((h, 20, 3))
        # if "diner_depth_guided_sampling" in self.cfg and self.cfg["diner_depth_guided_sampling"] and "one_mlp" in self.cfg and self.cfg["one_mlp"]:
        #     # , get_img('pixel_colors_nr'), gap]
        #     imgs = [get_img('pixel_colors_gt_fine'), gap]
        # else:
        #     if self.cfg["network"] == "neuray_ft":
        #         imgs = [get_img('pixel_colors_gt'), gap,
        #                 get_img('pixel_colors_nr'), gap]
        #         pass
        #     # network: neuray_ft
        #     else:
        #         # , get_img('pixel_colors_nr'), gap]
        #         imgs = [get_img('pixel_colors_gt'), gap]

        # # import ipdb; ipdb.set_trace()
        # if 'pixel_colors_dr' in data_pr: imgs.append(get_img('pixel_colors_dr'))
        imgs = []
        if 'pixel_colors_nr_fine' in data_pr:
            imgs.append(get_img('pixel_colors_nr_fine'))
        
        # if 'pixel_colors_dr_fine' in data_pr: imgs.append(get_img('pixel_colors_dr_fine'))
        data_index = kwargs['data_index']
        model_name = kwargs['model_name']
        # import ipdb;ipdb.set_trace()
        os.makedirs(f'data/vis_val/{model_name}/', exist_ok=True)
        imsave(
                f'data/vis_val/{model_name}/step-{step}-index-{data_index}-gt.jpg', concat_images_list(*[get_img('pixel_colors_gt')]))

        # # save images for final equi evaluation
        # Path(
        #     f'data/model/{model_name}/equi_evaluation').mkdir(exist_ok=True, parents=True)

        # imsave(f'data/model/{model_name}/equi_evaluation/step-{step}-index-{data_index}-pred.png',
        #        get_img('pixel_colors_nr_fine'))
        # imsave(f'data/model/{model_name}/equi_evaluation/step-{step}-index-{data_index}-gt.png',
        #        get_img('pixel_colors_gt'))
        # # import ipdb;ipdb.set_trace()

        Path(f'data/vis_val/{model_name}').mkdir(exist_ok=True, parents=True)
        if h <= 64 and w <= 64:
            imsave(
                f'data/vis_val/{model_name}/step-{step}-index-{data_index}.png', concat_images_list(*imgs))
        else:
            imsave(
                f'data/vis_val/{model_name}/step-{step}-index-{data_index}.jpg', concat_images_list(*imgs))
        
        # import ipdb;ipdb.set_trace()
        return outputs


name2metrics = {
    'psnr_ssim': PSNR_SSIM,
    'vis_img': VisualizeImage,
}


def psnr_nr(results):
    return np.mean(results['psnr_nr'])


def psnr_nr_fine(results):
    return np.mean(results['psnr_nr_fine'])


name2key_metrics = {
    'psnr_nr': psnr_nr,
    'psnr_nr_fine': psnr_nr_fine,
}
