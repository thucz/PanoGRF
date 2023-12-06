import time
import torch
import numpy as np
from tqdm import tqdm

from network.metrics import name2key_metrics
from train.train_tools import to_cuda


class ValidationEvaluator:
    def __init__(self,cfg):
        self.cfg = cfg
        self.key_metric_name=cfg['key_metric_name']
        self.key_metric=name2key_metrics[self.key_metric_name]

    def __call__(self, model, losses, eval_dataset, step, model_name, val_set_name=None):
        # import ipdb;ipdb.set_trace()
        if val_set_name is not None: model_name=f'{model_name}-{val_set_name}'

        model.eval()
        eval_results={}
        begin=time.time()
        for data_i, data in tqdm(enumerate(eval_dataset)):
            data = to_cuda(data)
            # import ipdb;ipdb.set_trace()
            if self.cfg["train_dataset_type"] == "gen":
                if data_i >= self.cfg["validate_num"]:
                    break;
            else:
                if data_i >= 2:
                    break;
            
            data['eval']=True
            data['step']=step
            with torch.no_grad():

                outputs=model(data)
                if self.cfg["debug"]:
                    import numpy as np
                    import cv2
                    import os
                    # import ipdb;ipdb.set_trace()
                    os.makedirs("./vis_bug/", exist_ok=True)

                    que_imgs_info = data["que_imgs_info"]
                    src_img=(que_imgs_info["imgs"].permute((0, 2, 3, 1)).data.cpu().numpy()[0]*255)
                    # src_img
                    cv2.imwrite("./vis_bug/src_img.jpg", src_img)
                    ref_imgs_info = data["ref_imgs_info"]
                    tgt_imgs= ref_imgs_info["imgs"].permute((0, 2, 3, 1)).data.cpu().numpy()
                    rfn = tgt_imgs.shape[0]
                    warp_pts = outputs["pts"].long().data.cpu().numpy() #2, rn, 2
                    warp_depths = outputs["depths"].data.cpu().numpy()#2,rn,1 
                    idx_1, idx_2 = np.where(warp_depths.squeeze(-1)==0)[0], np.where(warp_depths.squeeze(-1)==0)[1]
                    warp_pts[idx_1, idx_2, :] = np.asarray([0, 0]).reshape((1, 2))

                    # warp_depths
                    import ipdb;ipdb.set_trace()
                    warp_rgb = outputs["rgb"].data.cpu().numpy()#2, rn, 3
                    gt_depths = ref_imgs_info["depth"].squeeze(1).data.cpu().numpy()#1, 1, h, w
                    h, w = src_img.shape[:2]
                    final_depths=[]
                    
                    for rgb_idx in range(rfn):                    
                        # import ipdb;ipdb.set_trace()
                        depth_zero = np.zeros((h, w))
                        # gt_depths[rgb_idx]
                        depth_zero[warp_pts[rgb_idx, :, 1], warp_pts[rgb_idx, :, 0]] = \
                            warp_depths[rgb_idx].squeeze()                        
                        # depth_zero[torch.where()]
                        

                        final_depths.append(depth_zero)
                        cv2.imwrite("./vis_bug/warp_tgt_"+str(rgb_idx)+".jpg", np.uint8(warp_rgb[rgb_idx].reshape((h, w, 3))*255))
                        cv2.imwrite("./vis_bug/tgt_img_"+str(rgb_idx)+".jpg", np.uint8(tgt_imgs[rgb_idx]*255))                                                           
                    final_depths = np.asarray(final_depths)
                    def vis_depth(final_depths, gt_depths):
                        # import ipdb;ipdb.set_trace()
                        dmin=min(final_depths.min(), gt_depths.min())
                        dmax=min(final_depths.max(), gt_depths.max())
                        final_depths = np.uint8((final_depths-dmin)/(dmax-dmin)*255)
                        gt_depths =np.uint8((gt_depths-dmin)/(dmax-dmin)*255)
                        # color_depth_final = []
                        # color_depth_gt = []
                        # import ipdb;ipdb.set_trace()
    
                        for depth_idx in range(len(final_depths)):
                            # color_depth_final.append(cv2.applyColorMap(final_depths[depth_idx], cv2.COLORMAP_JET))
                            # color_depth_gt.append(cv2.applyColorMap(gt_depths[depth_idx], cv2.COLORMAP_JET))     
                            cv2.imwrite("./vis_bug/warp_depth_"+str(depth_idx)+".jpg", cv2.applyColorMap(final_depths[depth_idx], cv2.COLORMAP_JET))
                            cv2.imwrite("./vis_bug/gt_depth_"+str(depth_idx)+".jpg", cv2.applyColorMap(gt_depths[depth_idx], cv2.COLORMAP_JET))
                        
                    vis_depth(final_depths, gt_depths)
                    import ipdb;ipdb.set_trace()
                    
                #prj_depth = 
                #prj_rgb = 
                
                for loss in losses:
                    loss_results=loss(outputs, data, step, data_index=data_i, model_name=model_name)
                    for k,v in loss_results.items():
                        if type(v)==torch.Tensor:
                            v=v.detach().cpu().numpy()

                        if k in eval_results:
                            eval_results[k].append(v)
                        else:
                            eval_results[k]=[v]
        import numpy as np
        # import ipdb;ipdb.set_trace()
        for k,v in eval_results.items():
            eval_results[k]=np.concatenate(v,axis=0)

        key_metric_val=self.key_metric(eval_results)
        eval_results[self.key_metric_name]=key_metric_val
        print('eval cost {} s'.format(time.time()-begin))
        return eval_results, key_metric_val
