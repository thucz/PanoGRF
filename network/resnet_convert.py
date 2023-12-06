

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# ref1: https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736/3
# ref2: OmniFusion
def get_new_layer(m, bias=False):
    # new_layer = Conv2d(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride, \
    #     padding=m.padding, dilation=m.dilation, bias=bias)               
    # import pdb;pdb.set_trace()
   
    new_layer = nn.Conv3d(m.in_channels, m.out_channels, kernel_size=(m.kernel_size[0], m.kernel_size[1], 1), 
    stride=(m.stride[0], m.stride[1], 1), padding=(m.padding[0], m.padding[1], 0), padding_mode='zeros', bias=False)
    new_layer.weight.data.copy_(m.weight.data.unsqueeze(-1))
    if m.bias is not None:
        new_layer.bias.data.copy_(m.bias.data)

    new_layer.cuda()
    return new_layer

# # # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
# def get_new_bn(m):
    
#     if isinstance(m, nn.BatchNorm2d):
#         new_layer = nn.BatchNorm3d(m.num_features)
#     elif isinstance(m, nn.InstanceNorm2d):
#         new_layer = nn.InstanceNorm3d(m.num_features)

#     new_layer.weight.data.copy_(m.weight.data)
#     new_layer.bias.data.copy_(m.bias.data)
#     new_layer.running_mean.data.copy_(m.running_mean.data)
#     new_layer.running_var.data.copy_(m.running_var.data)
    
# #     new_maxpool=MaxPool2d(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation)
#     return new_layer

def setup_conv(target_attr):
    m = copy.deepcopy(target_attr)
    # import pdb;pdb.set_trace()
    if m.bias is not None:
        new_layer = get_new_layer(m, bias=True) 
        new_layer.bias.data.copy_(m.bias.data)
    else:
        new_layer = get_new_layer(m, bias=False) 
    new_layer.weight.data.copy_(m.weight.data.unsqueeze(-1))
    return new_layer

def setup_bn(target_attr):
    m = copy.deepcopy(target_attr)
    if isinstance(m, nn.BatchNorm2d):
        new_layer = nn.BatchNorm3d(m.num_features) # track_running_stats=False, affine=True
    elif isinstance(m, nn.InstanceNorm2d):
        new_layer = nn.InstanceNorm3d(m.num_features, track_running_stats = m.track_running_stats, affine = m.affine)
    # if 
    # if isinstance(new_layer, nn.BatchNorm2d):
    new_layer.weight.data.copy_(m.weight.data)
    new_layer.bias.data.copy_(m.bias.data)
   
    if isinstance(m, nn.BatchNorm2d):
        new_layer.running_mean.data.copy_(m.running_mean.data)
        new_layer.running_var.data.copy_(m.running_var.data)
    return new_layer
    


def replace_conv(module, name):
    if type(module) == torch.nn.Conv2d or isinstance(module, torch.nn.Conv2d):
        new_layer = setup_conv(module)   
        module = new_layer
        # import pdb;pdb.set_trace()
        return new_layer
    for n, ch in module.named_children():
        ch = replace_conv(ch, n)
        setattr(module, n, ch)
    return module

def replace_bn(module, name):
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
        # m = copy.deepcopy(sub_layer)
        new_layer = setup_bn(module)   
        module = new_layer
        # import pdb;pdb.set_trace()
        return new_layer
    for n, ch in module.named_children():
        ch = replace_bn(ch, n)
        setattr(module, n, ch)
    return module





def convert_to_tp(model):
    #b_list = [getattr(self.conv_stem, attr_str) for attr_str in dir(self.conv_stem)]     
    model = replace_conv(model, "net")
    model = replace_bn(model, "net")
    # model = replace_maxpool(model, 'net')
    # model = replace_conv3x3(model, 'net')
    return model