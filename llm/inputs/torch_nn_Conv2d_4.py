
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
    kernel_size = 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1
    bias = True
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.rand(1, in_channels, 32, 32).astype(dtype)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": bias,
        "padding_mode": padding_mode,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    in_channels = 1
    out_channels = 4
    kernel_size = 5
    stride = (2, 2)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 1
    bias = False
    padding_mode = 'zeros'
    dtype = np.float64
    input = np.random.rand(1, in_channels, 64, 64).astype(dtype)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": bias,
        "padding_mode": padding_mode,
        "dtype": torch.float64,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    in_channels = 3
    out_channels = 6
    kernel_size = 3
    stride = (1, 2)
    padding = (2, 1)
    dilation = (2, 1)
    groups = 3
    bias = True
    padding_mode = 'reflect'
    dtype = np.float32
    input = np.random.rand(2, in_channels, 28, 28).astype(dtype)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": bias,
        "padding_mode": padding_mode,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    in_channels = 4
    out_channels = 8
    kernel_size = 2
    stride = (3, 1)
    padding = (1, 0)
    dilation = (1, 2)
    groups = 2
    bias = False
    padding_mode = 'replicate'
    dtype = np.float64
    input = np.random.rand(1, in_channels, 16, 16).astype(dtype)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": bias,
        "padding_mode": padding_mode,
        "dtype": torch.float64,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    in_channels = 2
    out_channels = 4
    kernel_size = 4
    stride = (1, 1)
    padding = (2, 2)
    dilation = (1, 1)
    groups = 1
    bias = True
    padding_mode = 'circular'
    dtype = np.float32
    input = np.random.rand(4, in_channels, 24, 24).astype(dtype)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": bias,
        "padding_mode": padding_mode,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    in_channels = 5
    out_channels = 5
    kernel_size = 1
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 5
    bias = True
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.rand(1, in_channels, 10, 10).astype(dtype)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": bias,
        "padding_mode": padding_mode,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    in_channels = 7
    out_channels = 14
    kernel_size = 2
    stride = (2, 2)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 7
    bias = False
    padding_mode = 'reflect'
    dtype = np.float64
    input = np.random.rand(2, in_channels, 14, 14).astype(dtype)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": bias,
        "padding_mode": padding_mode,
        "dtype": torch.float64,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    in_channels = 9
    out_channels = 18
    kernel_size = 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = (2, 2)
    groups = 9
    bias = True
    padding_mode = 'replicate'
    dtype = np.float32
    input = np.random.rand(1, in_channels, 20, 20).astype(dtype)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": bias,
        "padding_mode": padding_mode,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    in_channels = 11
    out_channels = 22
    kernel_size = 4
    stride = (2, 1)
    padding = (2, 0)
    dilation = (1, 2)
    groups = 11
    bias = False
    padding_mode = 'circular'
    dtype = np.float64
    input = np.random.rand(4, in_channels, 22, 22).astype(dtype)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": bias,
        "padding_mode": padding_mode,
        "dtype": torch.float64,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    in_channels = 13
    out_channels = 13
    kernel_size = 1
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 13
    bias = True
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.rand(1, in_channels, 12, 12).astype(dtype)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": bias,
        "padding_mode": padding_mode,
        "dtype": torch.float32,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Conv2d_4"] = conv2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Conv2d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Conv2d_4'.")

check_valid('torch.nn.Conv2d', generated_inputs['torch.nn.Conv2d_4'], lib="torch", suffix=4)
