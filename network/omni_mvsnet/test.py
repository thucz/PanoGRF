#-1: monodepth使用inverse depth预测(在disparity空间监督训练);
#-2:-1, stereo depth也在disparity空间监督试试?

#todo:
#0. add uncertainty output for NeuRay
#1. 调整range
#2. 调整volume clamp min(done)
#3. add fixed_factor to tune sigma.
#4. 对比Uniform Depth Sampling(done)
#5. 对比只用单目指导的(wo_hdh)
#5.5 减小max_depth(改成15)(done)
#6.减少训练samples,(200000个太多了，太占空间了)(done: 20000)
#7. 同时测试 min_depth=0.1, min_depth=0.5

# for renderer避免过拟合: 修改 reference_idx看看warping 对不对，
# 随机化poses的点位，随机选que_id
# 对ft模型使用 depth-guided sampling(uncertainty-aware): mixed strategy

# iterations = 4. according to coarse cost-volume, input mono DEPTH and sigma and normalize, and process to output factor. (inaccuracy, scale and shift)
# 5. 先进行multi-view depth checking(单张图像深度根据Pose warp to other views,比较差异,根据差异tune sigma,
# ), 然后再生成cost-volume. 最后需要也生成sigma:(假设这个可以用loss监督?)
# 如果差异很大:
# 1).说明不准
# 2).也可能说明这个地方可能被遮挡了，不代表这里的单目深度是错的？



import torch
import numpy as np
#sampling_range = 3~8 <9
sampling_range = 5 #args["MAGNET_sampling_range"]        # beta in paper / defines the sampling range
# Uniform: 
n_samples=32
fixed_sigma=0.5 #5-4 = 1-0.5 = 2*sigma
# relaxation_factor=4
factor=2
fixed_sigma = factor*fixed_sigma#(mono)

# factor Learnable?



min_depth = 0.5
max_depth = 15.0
# clamp_min: min_depth
ref_mu = 5.0
n_depths=64
from scipy.special import erf
from scipy.stats import norm
P_total = erf(sampling_range / np.sqrt(2))             # Probability covered by the sampling range
idx_list = np.arange(0, n_samples + 1)
p_list = (1 - P_total)/2 + ((idx_list/n_samples) * P_total)
k_list = norm.ppf(p_list)
k_list = (k_list[1:] + k_list[:-1])/2
k_list = list(k_list)
print("k_list:", k_list)

depth_volume = [np.clip(ref_mu + k*fixed_sigma, a_min=1e-3, a_max=max_depth) for k in k_list]
print('depth_volume:', depth_volume)
# dv = 1.0 / torch.linspace(1 / min_depth, 1 / max_depth, n_depths)
dv = torch.linspace(min_depth, max_depth, n_depths)
print("dv:", dv)


