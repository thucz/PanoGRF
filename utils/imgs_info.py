import numpy as np
import torch
import math
from utils.base_utils import color_map_forward#, pad_img_end

# def random_crop(ref_imgs_info, que_imgs_info, target_size):
#     imgs = ref_imgs_info['imgs']
#     n, _, h, w = imgs.shape
#     out_h, out_w = target_size[0], target_size[1]
#     if out_w >= w or out_h >= h:
#         return ref_imgs_info

#     center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
#     center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)

#     def crop(tensor):
#         tensor = tensor[:, :, center_h - out_h // 2:center_h + out_h // 2,
#                               center_w - out_w // 2:center_w + out_w // 2]
#         return tensor

#     def crop_imgs_info(imgs_info):
#         imgs_info['imgs'] = crop(imgs_info['imgs'])
#         if 'depth' in imgs_info: imgs_info['depth'] = crop(imgs_info['depth'])
#         if 'true_depth' in imgs_info: imgs_info['true_depth'] = crop(imgs_info['true_depth'])
#         if 'masks' in imgs_info: imgs_info['masks'] = crop(imgs_info['masks'])

#         Ks = imgs_info['Ks'] # n, 3, 3
#         h_init = center_h - out_h // 2
#         w_init = center_w - out_w // 2
#         Ks[:,0,2]-=w_init
#         Ks[:,1,2]-=h_init
#         imgs_info['Ks']=Ks
#         return imgs_info

#     return crop_imgs_info(ref_imgs_info), crop_imgs_info(que_imgs_info)

# def random_flip(ref_imgs_info,que_imgs_info):
#     def flip(tensor):
#         tensor = np.flip(tensor.transpose([0, 2, 3, 1]), 2)  # n,h,w,3
#         tensor = np.ascontiguousarray(tensor.transpose([0, 3, 1, 2]))
#         return tensor

#     def flip_imgs_info(imgs_info):
#         imgs_info['imgs'] = flip(imgs_info['imgs'])
#         if 'depth' in imgs_info: imgs_info['depth'] = flip(imgs_info['depth'])
#         if 'true_depth' in imgs_info: imgs_info['true_depth'] = flip(imgs_info['true_depth'])
#         if 'masks' in imgs_info: imgs_info['masks'] = flip(imgs_info['masks'])

#         Ks = imgs_info['Ks']  # n, 3, 3
#         Ks[:, 0, :] *= -1
#         w = imgs_info['imgs'].shape[-1]
#         Ks[:, 0, 2] += w - 1
#         imgs_info['Ks'] = Ks
#         return imgs_info

#     ref_imgs_info = flip_imgs_info(ref_imgs_info)
#     que_imgs_info = flip_imgs_info(que_imgs_info)
#     return ref_imgs_info, que_imgs_info

# def pad_imgs_info(ref_imgs_info,pad_interval):
#     ref_imgs, ref_depths, ref_masks = ref_imgs_info['imgs'], ref_imgs_info['depth'], ref_imgs_info['masks']
#     ref_depth_gt = ref_imgs_info['true_depth'] if 'true_depth' in ref_imgs_info else None
#     rfn, _, h, w = ref_imgs.shape
    
#     ph = (pad_interval - (h % pad_interval)) % pad_interval
#     pw = (pad_interval - (w % pad_interval)) % pad_interval
    
#     if ph != 0 or pw != 0:
#         ref_imgs = np.pad(ref_imgs, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
#         ref_depths = np.pad(ref_depths, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
#         ref_masks = np.pad(ref_masks, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
#         if ref_depth_gt is not None:
#             ref_depth_gt = np.pad(ref_depth_gt, ((0, 0), (0, 0), (0, ph), (0, pw)), 'reflect')
#     ref_imgs_info['imgs'], ref_imgs_info['depth'], ref_imgs_info['masks'] = ref_imgs, ref_depths, ref_masks
#     if ref_depth_gt is not None:
#         ref_imgs_info['true_depth'] = ref_depth_gt
#     return ref_imgs_info

# def build_imgs_info(database, ref_ids, pad_interval=-1, is_aligned=True, align_depth_range=False, has_depth=True, replace_none_depth = False):    
#     ref_imgs = color_map_forward(np.asarray([database.get_image(ref_id) for ref_id in ref_ids])).transpose([0, 3, 1, 2])
#     ref_masks =  np.asarray([database.get_mask(ref_id) for ref_id in ref_ids], dtype=np.float32)[:, None, :, :]
#     if has_depth:
#         ref_depths = [database.get_depth(ref_id) for ref_id in ref_ids]
#         if replace_none_depth:
#             b, _, h, w = ref_imgs.shape
#             for i, depth in enumerate(ref_depths):
#                 if depth is None: ref_depths[i] = np.zeros([h, w], dtype=np.float32)
#         ref_depths = np.asarray(ref_depths, dtype=np.float32)[:, None, :, :]
#     else: ref_depths = None
#     ref_poses = np.asarray([database.get_pose(ref_id) for ref_id in ref_ids], dtype=np.float32)
#     # ref_Ks = np.asarray([database.get_K(ref_id) for ref_id in ref_ids], dtype=np.float32)
#     ref_depth_range = np.asarray([database.get_depth_range(ref_id) for ref_id in ref_ids], dtype=np.float32)
#     if align_depth_range:
#         ref_depth_range[:,0]=np.min(ref_depth_range[:,0])
#         ref_depth_range[:,1]=np.max(ref_depth_range[:,1])
#     ref_imgs_info = {'imgs': ref_imgs, 'poses': ref_poses, 'Ks': ref_Ks, 'depth_range': ref_depth_range, 'masks': ref_masks}
#     if has_depth: ref_imgs_info['depth'] = ref_depths
#     if pad_interval!=-1:
#         ref_imgs_info = pad_imgs_info(ref_imgs_info, pad_interval)
#     return ref_imgs_info


#Matterport3D
#before : has_depth =True
def build_imgs_info(database, ref_ids, pad_interval=-1, is_aligned=True, align_depth_range=False, has_depth=False, replace_none_depth = False, debug=False):
    # import ipdb;ipdb.set_trace()
    ref_imgs = np.asarray([database.get_image(ref_id) for ref_id in ref_ids]).transpose([0, 3, 1, 2])
    # ref_polar_weights = np.asarray([database.get_polar_weights(ref_id) for ref_id in ref_ids]).transpose([0, 3, 1, 2])
    # ref_polar_weights = database.get_polar_weights
    # height= self.cfg["height"]
    # width = self.cfg["width"]
    # batch_size = self.cfg["batch_size"]
    batch_size, _, height, width = ref_imgs.shape
    sin_phi = np.arange(0, height, dtype=np.float32) #.cuda()
    sin_phi = np.sin((sin_phi + 0.5) * math.pi / (height))
    #  = 
    ref_polar_weights = np.broadcast_to(sin_phi.reshape(1, 1, height, 1), (batch_size, 1, height, width)).copy()

    # self.sin_phi = sin_phi

    # import ipdb;ipdb.set_trace()# ref_imgs range: -> [0, 1]
    # print("ref_imgs.max(), min():", ref_imgs.max(), ref_imgs.min())
    ref_masks = np.asarray([database.get_mask(ref_id) for ref_id in ref_ids], dtype=np.float32)[:, None, :, :]
    if has_depth:
        ref_depths = [database.get_depth(ref_id) for ref_id in ref_ids]
        if replace_none_depth:
            b, _, h, w = ref_imgs.shape
            for i, depth in enumerate(ref_depths):
                if depth is None: ref_depths[i] = np.zeros([h, w], dtype=np.float32)
        ref_depths = np.asarray(ref_depths, dtype=np.float32)[:, None, :, :]
    else: ref_depths = None
    # ref_poses = np.asarray([database.get_pose(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_rots = np.asarray([database.get_rots(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_trans = np.asarray([database.get_trans(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_c2w = np.asarray([database.get_c2w(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_w2c = np.asarray([database.get_w2c(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_depth_range = np.asarray([database.get_depth_range(ref_id) for ref_id in ref_ids], dtype=np.float32)
    
    # ref_c2w = 

    #todo?
    # c2w = 
    # w2c = 

    # if align_depth_range:
    #     ref_depth_range[:,0]=np.min(ref_depth_range[:,0])
    #     ref_depth_range[:,1]=np.max(ref_depth_range[:,1])
    # , "rots": ref_rots, "trans": ref_trans
    ref_imgs_info = {'imgs': ref_imgs, "rots": ref_rots, "trans": ref_trans,  'c2w': ref_c2w, 'w2c': ref_w2c, 'depth_range': ref_depth_range, 'masks': ref_masks, 'polar_weights': ref_polar_weights}
    
    if has_depth: ref_imgs_info['depth'] = ref_depths
    # if pad_interval!=-1:
    #     ref_imgs_info = pad_imgs_info(ref_imgs_info, pad_interval)
    # if debug:
    #     import ipdb;ipdb.set_trace()
    return ref_imgs_info

def build_render_imgs_info(que_pose, que_shape,que_depth_range):
    h, w = que_shape
    h, w = int(h), int(w)
    que_coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1)
    que_coords = que_coords.reshape([1, -1, 2]).astype(np.float32)
    # ref_rots = np.asarray([database.get_rots(ref_id) for ref_id in ref_ids], dtype=np.float32)
    # ref_trans = np.asarray([database.get_trans(ref_id) for ref_id in ref_ids], dtype=np.float32)
    # ref_c2w = np.asarray([database.get_c2w(ref_id) for ref_id in ref_ids], dtype=np.float32)
    # ref_w2c = np.asarray([database.get_w2c(ref_id) for ref_id in ref_ids], dtype=np.float32)
    # ref_rots = 
    rots = que_pose[:3, :3]
    trans = que_pose[:3, 3]
    c2w = np.linalg.inv(np.concatenate([que_pose, np.array([[0, 0, 0, 1]])], axis=0))
    c2w = c2w[:3, :4]


    return {'w2c': que_pose.astype(np.float32)[np.newaxis,:,:],  # 1,3,4
            'c2w': c2w.astype(np.float32)[np.newaxis,...],
            'rots': rots.astype(np.float32)[np.newaxis,...],
            'trans': trans.astype(np.float32)[np.newaxis,...],            
            # 'Ks': que_K.astype(np.float32)[None,:,:],  # 1,3,3
            'coords': que_coords,
            'depth_range': np.asarray(que_depth_range, np.float32)[None, :],
            'shape': (h,w)}


# def build_render_cube_imgs_info(que_pose, que_shape,que_depth_range):
#     h, w = que_shape
#     h, w = int(h), int(w)
#     que_coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1)
#     que_coords = que_coords.reshape([1, -1, 2]).astype(np.float32)
#     # ref_rots = np.asarray([database.get_rots(ref_id) for ref_id in ref_ids], dtype=np.float32)
#     # ref_trans = np.asarray([database.get_trans(ref_id) for ref_id in ref_ids], dtype=np.float32)
#     # ref_c2w = np.asarray([database.get_c2w(ref_id) for ref_id in ref_ids], dtype=np.float32)
#     # ref_w2c = np.asarray([database.get_w2c(ref_id) for ref_id in ref_ids], dtype=np.float32)
#     # ref_rots = 
#     rots = que_pose[:3, :3]
#     trans = que_pose[:3, 3]
#     c2w = np.linalg.inv(np.concatenate([que_pose, np.array([[0, 0, 0, 1]])], axis=0))
#     c2w = c2w[:3, :4]


#     return {'w2c': que_pose.astype(np.float32)[np.newaxis,:,:],  # 1,3,4
#             'c2w': c2w.astype(np.float32)[np.newaxis,...],
#             'rots': rots.astype(np.float32)[np.newaxis,...],
#             'trans': trans.astype(np.float32)[np.newaxis,...],
            
#             # 'Ks': que_K.astype(np.float32)[None,:,:],  # 1,3,3
#             'coords': que_coords,
#             'depth_range': np.asarray(que_depth_range, np.float32)[None, :],
#             'shape': (h,w)}


def build_cube_imgs_info(database, ref_ids, pad_interval=-1, is_aligned=True, align_depth_range=False, has_depth=False, replace_none_depth = False, debug=False):
    ref_cube_imgs = np.asarray([database.get_cube_image(ref_id) for ref_id in ref_ids]).transpose([0, 3, 1, 2])
    batch_size, _, height, width = ref_cube_imgs.shape
    ref_cube_masks = np.asarray([database.get_cube_mask(ref_id) for ref_id in ref_ids], dtype=np.float32)[:, None, :, :]

# que_Ks = np.asarray([database.get_K(render_id) for render_id in render_ids],np.float32)
# que_poses = np.asarray([database.get_cube_w2c(render_id) for render_id in render_ids],np.float32)
    ref_Ks = np.asarray([database.get_K(ref_id) for ref_id in ref_ids],np.float32)
    ref_poses = np.asarray([database.get_cube_w2c(ref_id) for ref_id in ref_ids],np.float32)
    if has_depth:
        ref_depths = [database.get_cube_depth(ref_id) for ref_id in ref_ids]
        if replace_none_depth:
            b, _, h, w = ref_imgs.shape
            for i, depth in enumerate(ref_depths):
                if depth is None: ref_depths[i] = np.zeros([h, w], dtype=np.float32)
        ref_depths = np.asarray(ref_depths, dtype=np.float32)[:, None, :, :]
    else: ref_depths = None
    # ref_poses = np.asarray([database.get_pose(ref_id) for ref_id in ref_ids], dtype=np.float32)
    # ref_cube_rots = np.asarray([database.get_cube_rots(ref_id) for ref_id in ref_ids], dtype=np.float32)
    # ref_cube_trans = np.asarray([database.get_cube_trans(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_cube_c2w = np.asarray([database.get_cube_c2w(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_cube_w2c = np.asarray([database.get_cube_w2c(ref_id) for ref_id in ref_ids], dtype=np.float32)
    ref_depth_range = np.asarray([database.get_depth_range(ref_id) for ref_id in ref_ids], dtype=np.float32)
    
    # ref_c2w = 

    #todo?
    # c2w = 
    # w2c = 

    # if align_depth_range:
    #     ref_depth_range[:,0]=np.min(ref_depth_range[:,0])
    #     ref_depth_range[:,1]=np.max(ref_depth_range[:,1])
    # "cube_rots": ref_cube_rots, "cube_trans": ref_cube_trans
    ref_imgs_info = {'cube_imgs': ref_cube_imgs,  'cube_c2w': ref_cube_c2w, 'cube_w2c': ref_cube_w2c, 'depth_range': ref_depth_range, 'cube_masks': ref_cube_masks, 'Ks': ref_Ks, 'poses': ref_poses}
    if has_depth: ref_imgs_info['cube_depth'] = ref_depths
    # if pad_interval!=-1:
    #     ref_imgs_info = pad_imgs_info(ref_imgs_info, pad_interval)
    # if debug:
    #     import ipdb;ipdb.set_trace()
    return ref_imgs_info

def build_render_cube_imgs_info(que_pose,que_K,que_shape,que_depth_range):
    
    h, w = que_shape
    h, w = int(h), int(w)

    que_coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1)
    que_coords = que_coords.reshape([1, -1, 2]).astype(np.float32)
    return {'poses': que_pose.astype(np.float32)[None,:,:],  # 1,3,4
            'Ks': que_K.astype(np.float32)[None,:,:],  # 1,3,3
            'coords': que_coords,
            'depth_range': np.asarray(que_depth_range, np.float32)[None, :],
            'shape': (h,w)}


def imgs_info_to_torch(imgs_info):
    for k, v in imgs_info.items():
        if isinstance(v,np.ndarray):
            imgs_info[k] = torch.from_numpy(v).clone()
    return imgs_info

def imgs_info_slice(imgs_info, indices):
    imgs_info_out={}
    for k, v in imgs_info.items():
        imgs_info_out[k] = v[indices]
    return imgs_info_out
