
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv2d_inputs():
    list_of_inputs = []

    # Input 1
    in_channels = 3
    out_channels = 16
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 'valid'
    dilation = (1, 1)
    groups = 1
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float32
    input_tensor = torch.randn(1, in_channels, 32, 32).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    in_channels = 1
    out_channels = 4
    kernel_size = (5, 5)
    stride = (1, 1)
    padding = 'same'
    dilation = (1, 1)
    groups = 1
    bias = False
    padding_mode = 'zeros'
    dtype = torch.float64
    input_tensor = torch.randn(4, in_channels, 64, 64).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    in_channels = 3
    out_channels = 32
    kernel_size = (7, 7)
    stride = (1, 2)
    padding = 'valid'
    dilation = (2, 2)
    groups = 1
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float32
    input_tensor = torch.randn(1, in_channels, 128, 128).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    in_channels = 8
    out_channels = 16
    kernel_size = (3, 5)
    stride = (2, 1)
    padding = (1, 2)
    dilation = (1, 1)
    groups = 2
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float32
    input_tensor = torch.randn(2, in_channels, 64, 128).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    in_channels = 4
    out_channels = 12
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = 'same'
    dilation = (1, 1)
    groups = 4
    bias = False
    padding_mode = 'zeros'
    dtype = torch.float32
    input_tensor = torch.randn(1, in_channels, 16, 16).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    in_channels = 3
    out_channels = 16
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 'valid'
    dilation = (1, 1)
    groups = 1
    bias = True
    padding_mode = 'reflect'
    dtype = torch.float32
    input_tensor = torch.randn(1, in_channels, 32, 32).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    in_channels = 3
    out_channels = 16
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 'valid'
    dilation = (1, 1)
    groups = 1
    bias = True
    padding_mode = 'replicate'
    dtype = torch.float32
    input_tensor = torch.randn(1, in_channels, 32, 32).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    in_channels = 3
    out_channels = 16
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 'valid'
    dilation = (1, 1)
    groups = 1
    bias = True
    padding_mode = 'circular'
    dtype = torch.float32
    input_tensor = torch.randn(1, in_channels, 32, 32).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    in_channels = 3
    out_channels = 3
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 'valid'
    dilation = (1, 1)
    groups = 3
    bias = False
    padding_mode = 'zeros'
    dtype = torch.float32
    input_tensor = torch.randn(1, in_channels, 32, 32).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    in_channels = 1
    out_channels = 1
    kernel_size = (1, 1)
    stride = (1, 1)
    padding = 'valid'
    dilation = (1, 1)
    groups = 1
    bias = False
    padding_mode = 'zeros'
    dtype = torch.float32
    input_tensor = torch.randn(1, in_channels, 32, 32).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    in_channels = 1
    out_channels = 1
    kernel_size = (1, 1)
    stride = (1, 1)
    padding = 'valid'
    dilation = (1, 1)
    groups = 1
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float64
    input_tensor = torch.randn(1, in_channels, 32, 32).type(dtype).numpy()
    input_dict = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
                  'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups,
                  'bias': bias, 'padding_mode': padding_mode, 'dtype': dtype, 'input': input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Conv2d_7"] = conv2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Conv2d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Conv2d_7'.")

check_valid('torch.nn.Conv2d', generated_inputs['torch.nn.Conv2d_7'], lib="torch", suffix=7)
