import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

# from asset import example_scene_name2inter_ids, blended_mvs_ids
from dataset.database import BaseDatabase #, ExampleDatabase
from utils.base_utils import pose_inverse, transform_points_Rt


# def interpolate_render_poses(database, inter_img_ids, view_num, loop=True):    
#     # if loop:
#     #     inter_img_ids = list(inter_img_ids)+list(inter_img_ids[:-1:-1])
#         # inter_img_ids = np.append(inter_img_ids, inter_img_ids[0])
#     poses = [database.get_w2c(str(img_id)) for img_id in inter_img_ids]

#     # poses_inv = [pose_inverse(pose) for pose in poses]
    
#     # cam_pts = np.asarray(poses_inv)[:, :, 3]
#     # cam_rots = np.asarray([pose[:,:3] for pose in poses])

#     # rot_ang = []
#     # for k in range(len(inter_img_ids) - 1):
#     #     ang = np.linalg.norm(Rotation.from_matrix(cam_rots[k+1] @ cam_rots[k].T).as_rotvec())
#     #     rot_ang.append(ang)    
#     # rot_ang_sum = np.cumsum(np.asarray(rot_ang))
#     # rot_ang_sum = np.concatenate([np.zeros(1), rot_ang_sum],0)
#     # rot_ang_eval = np.linspace(rot_ang_sum[0],rot_ang_sum[-1],view_num)
#     # rotations = Rotation.from_matrix(cam_rots)
#     # import ipdb;ipdb.set_trace()
#     # rotations = Slerp(rot_ang_sum, rotations)(rot_ang_eval)
#     # rotations = rotations.as_matrix()
#     # translations = CubicSpline(rot_ang_sum,cam_pts)(rot_ang_eval)
#     # R = rotations
#     # t = rotations @ -translations[:,:,None]
#     return np.concatenate([R,t],2) # n,3,4

def interpolate_views(n_views_add, start_pose, end_pose):
    #3x4
    #R,T

    delta = (end_pose - start_pose)/(n_views_add+1)#
    new_poses_add = []
    for i in range(n_views_add):
        pose_add = start_pose + delta*(i+1)
        new_poses_add.append(pose_add)
    return new_poses_add
# def interpolate_poses(poses, n_poses):

def interpolate_render_poses(database, inter_img_ids, view_num, loop=True):    
    # if loop:
    #     inter_img_ids = list(inter_img_ids)+list(inter_img_ids[:-1:-1])
        # inter_img_ids = np.append(inter_img_ids, inter_img_ids[0])
    poses = [database.get_w2c(str(img_id)) for img_id in inter_img_ids]

    #在已有的poses中进行插值，假设输入的poses按照固定的拍摄顺序
    add_poses_len = view_num - len(poses)#58

    add = add_poses_len // (len(poses)-1)#58
    rest = add_poses_len % (len(poses)-1)#0
    print('add, rest:', add, rest)
    new_poses = []
    #poses[i]->poses[i+1]
    for i in range(len(poses)-1):
        # i, i+1
        # interpolate views
        if i < rest:
        #     #poses[i]和poses[i+1]的pose
            add_poses = interpolate_views(add+1, poses[i], poses[i+1])
        else:
            add_poses = interpolate_views(add, poses[i], poses[i+1])
        new_poses.append(poses[i])
        new_poses+=add_poses
    new_poses.append(poses[-1])#last pose
    new_poses = np.array(new_poses)

    return new_poses

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec2, vec1_avg))
    vec1 = normalize(np.cross(vec0, vec2))
    m = np.stack([-vec0, vec1, vec2, pos], 1)
    return m

def render_path_axis(c2w, up, ax, rad, focal, N):
    render_poses = []
    center = c2w[:, 3]
    v = c2w[:, ax] * rad
    for t in np.linspace(-1., 1., N + 1)[:-1]:
        c = center + t * v
        z = normalize((center + focal * c2w[:, 2]) - c)
        render_poses.append(np.concatenate([viewmatrix(z, up, c)], 1))
    return render_poses

def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([-np.sin(theta), np.cos(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.])) - c)
        render_poses.append(np.concatenate([viewmatrix(z, up, c)], 1))
    return render_poses

def forward_circle_poses(database:BaseDatabase):
    poses = [database.get_pose(img_id) for img_id in database.get_img_ids()]
    poses_inv = [pose_inverse(pose) for pose in poses]
    cam_pts = np.asarray(poses_inv)[:, :, 3]
    cam_rots = np.asarray(poses_inv)[:, :, :3]
    down = cam_rots[:, :, 1]
    lookat = cam_rots[:, :, 2]

    avg_cam_pt = (np.max(cam_pts,0)+np.min(cam_pts,0))/2
    avg_down = np.mean(down,0)
    avg_lookat = np.mean(lookat,0)
    avg_pose_inv = viewmatrix(avg_lookat, avg_down, avg_cam_pt)
    avg_pose = pose_inverse(avg_pose_inv)

    cam_pts_in_avg_pose = transform_points_Rt(cam_pts,avg_pose[:,:3],avg_pose[:,3]) # n,3
    range_in_avg_pose = np.percentile(np.abs(cam_pts_in_avg_pose), 90, 0)

    depth_ranges = [database.get_depth_range(img_id) for img_id in database.get_img_ids()]
    depth_ranges = np.asarray(depth_ranges)
    near, far = np.mean(depth_ranges[:,0]), np.mean(depth_ranges[:,1])
    dt = .75
    mean_dz = 1. / (((1. - dt) / near + dt / far))
    z_delta = near * 0.2
    range_in_avg_pose[2] = z_delta
    shrink_ratio = 0.8
    range_in_avg_pose*=shrink_ratio

    render_poses=render_path_spiral(avg_pose_inv,avg_down,range_in_avg_pose,mean_dz,0.,1,60)
    render_poses=[pose_inverse(pose) for pose in render_poses]
    render_poses=np.asarray(render_poses)
    return render_poses

def interpolate_poses(database: BaseDatabase):
    img_ids = [0, 2]
    que_poses = interpolate_render_poses(database, img_ids, 120, False)
    
    # else:
    #     raise NotImplementedError
    return que_poses

def get_render_poses(database, pose_type, pose_fn=None):
    if pose_type.startswith('inter'):
        que_poses = interpolate_poses(database)
        # inter_num = int(pose_type.split('_')[1])
        # inter_ids = np.loadtxt(pose_fn, dtype=np.int64)
        # inter_ids = np.asarray(database.get_img_ids())[inter_ids]
        # que_poses = interpolate_render_poses(database, inter_ids, inter_num)
    # elif pose_type=='circle':
    #     que_poses = forward_circle_poses(database)
    else:
        raise NotImplementedError
    return que_poses

def get_render_cube_poses(database, pose_type, pose_fn=None, cube_id=1):
    if pose_type.startswith('inter'):
        # que_poses = interpolate_poses(database)
        # inter_num = int(pose_type.split('_')[1])
        # inter_ids = np.loadtxt(pose_fn, dtype=np.int64)
        # inter_ids = np.asarray(database.get_img_ids())[inter_ids]
        # que_poses = interpolate_render_poses(database, inter_ids, inter_num)
        que_poses = database.get_cube_interpolate_render_poses(inter_num=120, cube_id=cube_id)
    elif pose_type=='circle':
        que_poses = forward_circle_poses(database)
    else:
        raise NotImplementedError
    return que_poses