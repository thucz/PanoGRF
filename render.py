import argparse
import os
from pathlib import Path
import imageio

import numpy as np
import torch
from skimage.io import imsave
from tqdm import tqdm
from dataset.database import  M3DDatabase#, ResidentialDatabase #, CoffeeAreaDatabase
# from data_readers.replica_wide import ReplicaWideDataset
# from data_readers.residential import ResidentialDataset
# from data_readers.coffeearea import CoffeeAreaDataset
# from dataset.database import parse_database_name, get_database_split, ExampleDatabase
from dataset.database import get_database_split
# from dataset.train_dataset import build_src_imgs_info_select
from network.renderer import name2network
from utils.base_utils import load_cfg, to_cuda, color_map_backward, make_dir
from utils.imgs_info import build_imgs_info, build_render_imgs_info, build_render_cube_imgs_info, imgs_info_to_torch, imgs_info_slice
from utils.render_poses import get_render_poses
# from utils.view_select import select_working_views_db

def prepare_render_info(database, pose_type, pose_fn, use_depth):
    # interpolate poses
    if pose_type.startswith('eval'):#todo
        split_name = 'test' # else 'test_all'
        ref_ids, render_ids = get_database_split(database, split_name)
        # que_Ks = np.asarray([database.get_K(render_id) for render_id in render_ids],np.float32)
        #w2c
        que_poses = np.asarray([database.get_w2c(render_id) for render_id in render_ids],np.float32)
        que_shapes = np.asarray([database.get_image(render_id).shape[:2] for render_id in render_ids],np.int64)
        que_depth_ranges = np.asarray([database.get_depth_range(render_id) for render_id in render_ids],np.float32)

    elif pose_type.startswith('inter'):#done
        que_poses = get_render_poses(database, pose_type, pose_fn)
        # import ipdb;ipdb.set_trace()
        # prepare intrinsics, shape, depth range
        # que_Ks = np.array([database.get_K(database.get_img_ids()[0]) for _ in range(que_poses.shape[0])],np.float32)
        h, w, _ = database.get_image(database.get_img_ids()[0]).shape
        que_shapes = np.array([(h,w) for _ in range(que_poses.shape[0])])
        # if isinstance(database,ExampleDatabase):
        #     # we have sparse points to compute depth range
        #     que_depth_ranges = np.stack([database.compute_depth_range_impl(pose) for pose in que_poses],0)
        # else:
        # just use depth range of all images
        ref_depth_range_list = np.asarray([database.get_depth_range(img_id) for img_id in database.get_img_ids()])        
        near = np.min(ref_depth_range_list[:,0])
        far = np.max(ref_depth_range_list[:,1])
        que_depth_ranges = np.asarray([(near,far) for _ in range(que_poses.shape[0])],np.float32)
        ref_ids = [0, 2]#database.get_img_ids()
        render_ids = None
    else:
        print("input correct pose_type")
        raise Exception
    return que_poses,  que_shapes, que_depth_ranges, ref_ids, render_ids

def save_renderings(output_dir, qi, render_info, h, w):
    def output_image(suffix):
        if f'pixel_colors_{suffix}' in render_info:
            render_image = color_map_backward(render_info[f'pixel_colors_{suffix}'].cpu().numpy().reshape([h, w, 3]))
            imsave(f'{output_dir}/{qi}-{suffix}.jpg', render_image)
        return render_image
    # output_image('nr')
    fine_image = output_image('nr_fine')
    return fine_image

import cv2
def save_depth(output_dir, qi, render_info, h, w, depth_range, gt_depth=None):
    suffix='fine'
    if f'render_depth_{suffix}' in render_info:
        near, far = depth_range
        depth = render_info[f'render_depth_{suffix}'].cpu().numpy().reshape([h, w])
        # import ipdb;ipdb.set_trace()
        depth = np.clip(depth, a_min=near, a_max=far)
        depth = (1/depth - 1/near)/(1/far - 1/near)
        # depth = color_map_backward(depth)
        # depth = 
        # d_min = np.min(depth)
        # d_max = np.max(depth)
        # import ipdb;ipdb.set_trace()
        # d_norm = np.uint8((depth-d_min)/(d_max-d_min)*255)
        d_norm = np.uint8(depth*255)
            
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)        
        imsave(f'{output_dir}/{qi}-{suffix}-depth.png', d_color)
        if gt_depth is not None:
            gt_depth = np.clip(gt_depth, a_min=near, a_max=far)
            gt_depth = (1/gt_depth - 1/near)/(1/far - 1/near)
            gt_d_norm = np.uint8(gt_depth*255)        
            gt_d_color = cv2.applyColorMap(gt_d_norm, cv2.COLORMAP_JET)
            imsave(f'{output_dir}/{qi}-{suffix}-gt-depth.png', gt_d_color)

        # if gt_depth is not None:
        #     pass


def render_video_gen(database_name: str,
                     cfg_fn='configs/gen_lr_neuray.yaml',
                     pose_type='inter', pose_fn=None,
                     render_depth=False,
                     ray_num=8192, rb=0, re=-1, data_idx=0, m3d_dist=0.5):
    default_cfg={
        "MAGNET_mvs_weighting": "CW5",
        "wo_hdh": False,
        "change_input": False,
        "revise_range": False,
        "handle_distort": False,
        "handle_distort_all": False,
        "handle_distort_input_all": False,
        "use_polar_weighted_loss": False,
        "eval_only": False,
        "render_uncert": False,
        "uncert_tune": False,
        "use_disp": True,
        "with_sin": False,
        "wo_mono_feat": False,
        "mono_uncert_tune": False,
        "fix_all": False,
        "fix_coarse": False,  
        "use_depth": False, 

   }
    # cfg = load_cfg(cfg_fn)
    cfg = {**default_cfg, **load_cfg(cfg_fn)}

    # load render cfg
    # cfg = load_cfg(cfg_fn)
    cfg['ray_batch_num'] = ray_num
    cfg["m3d_dist"] = m3d_dist
    # render_cfg = cfg['train_dataset_cfg'] if 'train_dataset_cfg' in cfg else {}
    # render_cfg = {**default_render_cfg , **cfg}
    # render_cfg = cfg
    cfg['render_depth'] = render_depth
    cfg['use_depth'] = False #default_render_cfg['use_depth']
    cfg['render_uncert'] = False
    # render_cfg = cfg

    # cfg['']
    if database_name == "residential":
        cfg["dataset_name"] = database_name
    elif database_name in ["m3d", "replica_wide"]:
        cfg["dataset_name"] = "m3d"
    elif database_name in ["CoffeeArea"]:
        cfg["dataset_name"] = database_name
    else:
        raise Exception
    # load model
    renderer = name2network[cfg['network']](cfg)

    ckpt = torch.load(f'data/model/{cfg["name"]}/model.pth')
    renderer.load_state_dict(ckpt['network_state_dict'])
    renderer.cuda()
    renderer.eval()
    step = ckpt["step"]

    if database_name == "replica_wide":
        # mode="test"
        test_set = ReplicaWideDataset(
            cfg=cfg
        )
    
        data = test_set.__getitem__(data_idx)
        # data = dataset.__getitem__(1)
        # import ipdb;ipdb.set_trace()
        database = M3DDatabase(cfg, data)
        
    elif database_name in ["CoffeeArea"]:
        test_set = CoffeeAreaDataset(
            cfg=cfg
        )
        data = test_set.__getitem__(data_idx)
        # data = dataset.__getitem__(1)
        # import ipdb;ipdb.set_trace()
        database = CoffeeAreaDatabase(cfg, data)
    elif database_name in ["m3d"]:

        if cfg["use_lmdb"]:
            from data_readers.habitat_data_neuray_ft_lmdb import HabitatImageGeneratorFT_LMDB
            mode="test"
            test_set = HabitatImageGeneratorFT_LMDB(
                args=cfg,
                split=mode,
                seq_len=cfg["seq_len"],
                reference_idx=cfg["reference_idx"],
                full_width=cfg["width"],
                full_height=cfg["height"],
                m3d_dist=cfg["m3d_dist"]
            )

        else:
            from data_readers.habitat_data_neuray_ft import HabitatImageGeneratorFT
                
            mode="test"
            test_set = HabitatImageGeneratorFT(
                args=cfg,
                split=mode,
                seq_len=cfg["seq_len"],
                reference_idx=cfg["reference_idx"],
                full_width=cfg["width"],
                full_height=cfg["height"],
                m3d_dist=cfg["m3d_dist"]
            )
        data = test_set.__getitem__(data_idx)
        # data = dataset.__getitem__(1)
        # import ipdb;ipdb.set_trace()
        database = M3DDatabase(cfg, data)
    elif database_name == "residential":
        # cfg["dataset_name"] = "residential"
        test_set = ResidentialDataset(
            cfg=cfg
        )
        data = test_set.__getitem__(data_idx)
        # data = dataset.__getitem__(1)
        # import ipdb;ipdb.set_trace()
        database = ResidentialDatabase(cfg, data)
    else:
        raise Exception
    
    # database = database#parse_database_name(self.cfg['database_name'])    
    que_poses, que_shapes, que_depth_ranges, ref_ids_all, render_ids = \
        prepare_render_info(database, pose_type, pose_fn, cfg['use_depth'])
    # import ipdb;ipdb.set_trace()

    # select working views
    # overlap_select = False
    # if overlap_select:
    #     ref_ids_list = []
    #     ref_size = database.get_image(ref_ids_all[0]).shape[:2]
    #     ref_poses = np.stack([database.get_pose(ref_id) for ref_id in ref_ids_all], 0)
    #     ref_Ks = np.stack([database.get_K(ref_id) for ref_id in ref_ids_all], 0)
    #     for que_pose, que_K, que_shape, que_depth_range in zip(que_poses, que_Ks, que_shapes, que_depth_ranges):
    #         ref_indices = select_working_views_by_overlap(ref_poses, ref_Ks, ref_size, que_pose, que_K, que_shape, que_depth_range, render_cfg['min_wn'])
    #         ref_ids_list.append(np.asarray(ref_ids_all)[ref_indices])
    # else:
    # ref_ids_list = select_working_views_db(database, ref_ids_all, que_poses, render_cfg['min_wn'])
    # import ipdb;ipdb.set_trace()
    output_dir = f'data/render/{database_name}_{cfg["m3d_dist"]}/{cfg["name"]}-{step}-{pose_type}-{data_idx}'
    if "freeview" in cfg and cfg["freeview"]:
        output_dir += "_freeview_"+str(cfg["offset"][0])+","+str(cfg["offset"][1])+","+str(cfg["offset"][2])
    
    # if overlap_select: output_dir+='-overlap'
    make_dir(output_dir)
    # import ipdb;ipdb.set_trace()
    # render
    num = que_poses.shape[0]
    re = num if re==-1 else re
    print("rb, re:", rb, re)
    imgs = []    
    for qi in tqdm(range(rb,re)):
        if os.path.exists(f'{output_dir}/{qi}-nr_fine.jpg'): 
            ret_img = cv2.imread(f'{output_dir}/{qi}-nr_fine.jpg')
            ret_img = ret_img[..., ::-1]
            imgs.append(ret_img)
            continue
        que_imgs_info = build_render_imgs_info(que_poses[qi], que_shapes[qi], que_depth_ranges[qi])

        que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
        data_contain_all = False
        if data_contain_all:
            data["ref_imgs_info"] = to_cuda(data["ref_imgs_info"])
            data["src_imgs_info"] = to_cuda(data["src_imgs_info"])
            data['que_imgs_info'] = to_cuda(que_imgs_info) 
        else:
            data = {'que_imgs_info': que_imgs_info, 'eval': True}
            ref_ids = ref_ids_all #list[qi]
            ref_imgs_info = build_imgs_info(database, ref_ids)#?
            src_ids = [2, 0]#
            src_imgs_info = build_imgs_info(database, src_ids)#?
            ref_imgs_info = to_cuda(imgs_info_to_torch(ref_imgs_info))
            data['ref_imgs_info']=ref_imgs_info
            data['src_imgs_info'] = to_cuda(imgs_info_to_torch(src_imgs_info))       


    
        # que_imgs_info = build_render_imgs_info(que_poses[qi], que_shapes[qi], que_depth_ranges[qi])
        # que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
        with torch.no_grad():
            render_info = renderer(data)
        h, w = que_shapes[qi]        
        ret_img = save_renderings(output_dir, qi, render_info, h, w)
        # import ipdb;ipdb.set_trace()

        if render_depth:
            if pose_type=='eval':
                test_view=1
                gt_depth = database.get_depth(test_view)#middle
                save_depth(output_dir, qi, render_info, h, w, que_depth_ranges[qi], gt_depth=gt_depth)
            else:
                save_depth(output_dir, qi, render_info, h, w, que_depth_ranges[qi])
            
        imgs.append(ret_img)
        if pose_type=='eval':
            # {database_name}_{cfg["m3d_dist"]}/{cfg["name"]}-{step}-{pose_type}-{data_idx}
            gt_dir = f'data/render/{database_name}_{cfg["m3d_dist"]}/{cfg["name"]}-{step}-{pose_type}-{data_idx}-gt'
            if "freeview" in cfg and cfg["freeview"]:
                gt_dir += "_freeview_"+str(cfg["offset"][0])+","+str(cfg["offset"][1])+","+str(cfg["offset"][2])
            
            Path(gt_dir).mkdir(exist_ok=True, parents=True)
            if not (Path(gt_dir)/f'{qi}.jpg').exists():
                imsave(f'{gt_dir}/{qi}.jpg',database.get_image(render_ids[qi]))
    if pose_type=='eval':
        pass
    else:
        imageio.mimsave(f'{output_dir}/nr_fine.gif', imgs, fps=30)
        

def render_video_ft(database_name, cfg_fn, pose_type, pose_fn, render_depth=False, ray_num=4096, rb=0, re=-1, data_idx=0, m3d_dist=0.5):
    # init network    
    default_cfg={
        "MAGNET_mvs_weighting": "CW5",
        "wo_hdh": False,
        "change_input": False,
        "revise_range": False,
        "handle_distort": False,
        "handle_distort_all": False,
        "handle_distort_input_all": False,
        "use_polar_weighted_loss": False,
        "eval_only": False,
        "render_uncert": False,
        "uncert_tune": False,
        "use_disp": True,
        "with_sin": False,
        "wo_mono_feat": False,
        "mono_uncert_tune": False,
        "fix_all": False,
        "fix_coarse": False,  
        "use_depth": False, 
    }
    # cfg = load_cfg(cfg_fn)
    cfg = {**default_cfg, **load_cfg(cfg_fn)}
    
    # import ipdb;ipdb.set_trace()
    
    if cfg["train_dataset_type"] == "gen":
        pass
    else:
        cfg["data_idx"] = data_idx
        cfg["name"] = cfg["name"]+"_id_"+str(data_idx)
    
    # cfg['gen_cfg'] = None
    cfg['validate_initialization'] = False
    cfg['ray_batch_num'] = ray_num
    cfg['render_depth'] = False #render_depth
    cfg['render_uncert'] = False
    ckpt = torch.load(f'data/model/{cfg["name"]}/model.pth')
    _, dim, h, w = ckpt['network_state_dict']['ray_feats.0'].shape
    cfg['ray_feats_res'] = [h,w]
    cfg['ray_feats_dim'] = dim
    renderer = name2network[cfg['network']](cfg)
    renderer.load_state_dict(ckpt['network_state_dict'])
    step=ckpt['step']
    renderer.cuda()
    renderer.eval()
    
    #todo
    # database = parse_database_name(database_name)
    database = renderer.database

    # database
    que_poses, que_shapes, que_depth_ranges, ref_ids, render_ids = \
        prepare_render_info(database, pose_type, pose_fn, False)
    
    # assert(database.database_name == renderer.database.database_name)
    output_dir = f'data/render/{database_name}_{m3d_dist}/{cfg["name"]}-{step}-{pose_type}'
    
    
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    if pose_type == "eval":
        gt_output_dir = f'data/render/{database_name}_{m3d_dist}/{cfg["name"]}-{step}-{pose_type}-gt'
        Path(gt_output_dir).mkdir(parents=True,exist_ok=True)
    
    # import ipdb;ipdb.set_trace()

    # render
    num = que_poses.shape[0]
    # import ipdb;ipdb.set_trace()
    re = num if re==-1 else re
    imgs = []
    for qi in tqdm(range(rb,re)):
        if os.path.exists(f'{output_dir}/{qi}.jpg'): continue
        que_imgs_info = build_render_imgs_info(que_poses[qi], que_shapes[qi], que_depth_ranges[qi])
        que_imgs_info = to_cuda(imgs_info_to_torch(que_imgs_info))
        with torch.no_grad():
            render_info = renderer.render_pose(que_imgs_info)
        h, w = que_shapes[qi]
        ret_img = save_renderings(output_dir, qi, render_info, h, w)
        imgs.append(ret_img)
        if render_depth:
            save_depth(output_dir, qi, render_info, h, w, que_depth_ranges[qi])

        if pose_type=='eval':
            # gt_dir = f'data/render/{database_name}/gt'
            # Path(gt_dir).mkdir(exist_ok=True, parents=True)
            # if not (Path(gt_dir)/f'{qi}.jpg').exists():
            imsave(f'{gt_output_dir}/{qi}.jpg',database.get_image(render_ids[qi]))
    
    # f'{output_dir}/{qi}-{suffix}.jpg'
    if pose_type == "eval":
        pass
    else:
        imageio.mimsave(f'{output_dir}/nr_fine.gif', imgs, fps=30)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_name', type=str, default='m3d', help='<dataset_name>/<scene_name>/<scene_setting>')
    parser.add_argument('--cfg', type=str, default='configs/train/ft/neuray_gen_cost_volume_train_erp.yaml', help='config path of the renderer')
    parser.add_argument('--pose_type', type=str, default='eval', help='type of render poses')    
    parser.add_argument('--pose_fn', type=str, default=None, help='file to render poses')
    parser.add_argument('--rb', type=int, default=0, help='begin index of rendering poses')
    parser.add_argument('--re', type=int, default=-1, help='end index of rendering poses')
    parser.add_argument('--render_type', type=str, default='gen', help='gen:generalization or ft:finetuning')
    parser.add_argument('--ray_num', type=int, default=4096, help='number of rays in one rendering batch')
    parser.add_argument('--depth', action='store_true', dest='depth', default=False)
    parser.add_argument('--data_idx', type=int, default=0, help='data index')
    parser.add_argument('--m3d_dist', type=float, default=0.5, help='data dist')

    # parser.add_argument('--overlap', action='store_true', dest='overlap', default=False)
    flags = parser.parse_args()
    # import ipdb;ipdb.set_trace()
    if flags.render_type=='gen':
        render_video_gen(flags.database_name, cfg_fn=flags.cfg, pose_type=flags.pose_type, pose_fn=flags.pose_fn,
                         render_depth=flags.depth, ray_num=flags.ray_num, rb=flags.rb,re=flags.re, data_idx=flags.data_idx , m3d_dist=flags.m3d_dist)
    else:
        render_video_ft(flags.database_name, cfg_fn=flags.cfg, pose_type=flags.pose_type, pose_fn=flags.pose_fn,
                        render_depth=flags.depth, ray_num=flags.ray_num, rb=flags.rb, re=flags.re, data_idx=flags.data_idx, m3d_dist=flags.m3d_dist)