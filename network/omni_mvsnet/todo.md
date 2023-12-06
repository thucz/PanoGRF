#0. train_seq_len 置为1， reference_idx置为0， idx置为0
#1. midfeature 换成x_d3, 然后可能还需要修改torch.sort()



1. mono_stereo 采样时同时加上32+5~10(保证单目不准时也可以用其他深度假设)
2. mono_stereo 采用depth的加权，利用uncertainty作为指导
3. FNET pretraining
4. 训练一个只用mono_depth的网络（无uncertainty),再mono_depth周围采样+32等disparity




revise:

add fnet

volume 32+mono32, remove multi-view checking

