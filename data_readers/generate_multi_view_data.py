import numpy as np
#todo: reference_idx = 1->4
#seq_len:5

# def generate_mv_4(rand_location, orig_location, m3d_dist, idx):
#     movement_deltas1 = {
#         0: [-m3d_dist, -m3d_dist],
#         1: [-m3d_dist, m3d_dist],
#         2: [m3d_dist, m3d_dist],
#         3: [m3d_dist, -m3d_dist]
#     }
#     # movement_deltas = {
#     #     0: m3d_dist,
#     #     1: 0.0,
#     #     2: -m3d_dist
#     # }
#     #z    
#     # rand_location[[2]] = (
#     #     orig_location[[2]] + movement_deltas[idx]
#     # )
#     if idx==len(movement_deltas1.keys()):
#         return rand_location
#     if idx >= 0 and idx < 4:
#         # change x, z
#         rand_location[0] += movement_deltas1[idx][0]
#         rand_location[2] += movement_deltas1[idx][1]
#     # elif
#     else:
#         raise ValueError("No implementation for rand location idx > 4")
#     return rand_location
            # num_views, self.reference_idx, rand_location, orig_location, self.m3d_dist, i, vertical_movements, outbound_movements, movements, working_views
def generate_mv(num_views, reference_idx, rand_location, orig_location, m3d_dist, iter_idx, vertical_movements, outbound_movements, movements, working_views):
    
    # movement_deltas1 = {
    #     0: [-m3d_dist, -m3d_dist],
    #     1: [-m3d_dist, m3d_dist],
    #     2: [m3d_dist, m3d_dist],
    #     3: [m3d_dist, -m3d_dist]
    # }


    if iter_idx==working_views: #len(movements.keys()):
        return rand_location
    elif iter_idx >= 0 and iter_idx < working_views: #iter
        # change x, z
        rand_location[0] += movements[iter_idx][0]
        rand_location[2] += movements[iter_idx][1]
    elif iter_idx > working_views:        
        # assert iter_idx-working_views<=max(outbound_movements.keys())        
        if iter_idx-working_views<=max(vertical_movements.keys()):
            rand_location[1] += vertical_movements[iter_idx-working_views] #y direction
        # elif iter_idx-working_views>max(vertical_movements.keys()) and iter_idx-working_views<=max(outbound_movements.keys()):
        #     rand_location[0] += outbound_movements[iter_idx-working_views] #x direction        
    else:
        raise ValueError("No implementation for rand location idx > working views")

    return rand_location
        
"""
renderer
x|y?
            11
    1               2
            5
            
14      8   0   6       12 ->z
        
            7
    3               4 
            13
(9/10)
       
P0: (0, 0, 0)
P1: (0-m3d_dist)
depth:
1               2

        0(no need)

3               4 
"""





