
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv2d_inputs():
    list_of_inputs = []

    # Input 1
    in_channels = np.int32(3)
    out_channels = np.int32(16)
    kernel_size = (np.int32(3), np.int32(3))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(1), np.int32(1))
    dilation = (np.int32(1), np.int32(1))
    groups = np.int32(1)
    bias = True
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.rand(1, 3, 32, 32).astype(dtype)

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
    in_channels = np.int32(1)
    out_channels = np.int32(4)
    kernel_size = (np.int32(5), np.int32(5))
    stride = (np.int32(2), np.int32(2))
    padding = (np.int32(0), np.int32(0))
    dilation = (np.int32(1), np.int32(1))
    groups = np.int32(1)
    bias = False
    padding_mode = 'zeros'  # Changed from 'valid'
    dtype = np.float64
    input = np.random.rand(1, 1, 64, 64).astype(dtype)

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
    in_channels = np.int32(3)
    out_channels = np.int32(32)
    kernel_size = (np.int32(3), np.int32(5))
    stride = (np.int32(1), np.int32(2))
    padding = (np.int32(2), np.int32(4))
    dilation = (np.int32(2), np.int32(1))
    groups = np.int32(1)
    bias = True
    padding_mode = 'reflect'
    dtype = np.float32
    input = np.random.rand(1, 3, 28, 28).astype(dtype)

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
    in_channels = np.int32(64)
    out_channels = np.int32(128)
    kernel_size = (np.int32(1), np.int32(1))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(0), np.int32(0))
    dilation = (np.int32(1), np.int32(1))
    groups = np.int32(1)
    bias = False
    padding_mode = 'zeros'
    dtype = np.float64
    input = np.random.rand(1, 64, 16, 16).astype(dtype)

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
    in_channels = np.int32(3)
    out_channels = np.int32(6)
    kernel_size = (np.int32(7), np.int32(7))
    stride = (np.int32(4), np.int32(4))
    padding = (np.int32(3), np.int32(3))
    dilation = (np.int32(1), np.int32(1))
    groups = np.int32(3)
    bias = True
    padding_mode = 'replicate'
    dtype = np.float32
    input = np.random.rand(1, 3, 128, 128).astype(dtype)

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
    in_channels = np.int32(16)
    out_channels = np.int32(32)
    kernel_size = (np.int32(2), np.int32(4))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(1), np.int32(2))
    dilation = (np.int32(1), np.int32(1))
    groups = np.int32(1)
    bias = True
    padding_mode = 'circular'
    dtype = np.float32
    input = np.random.rand(1, 16, 32, 64).astype(dtype)

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
    in_channels = np.int32(1)
    out_channels = np.int32(1)
    kernel_size = (np.int32(1), np.int32(1))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(0), np.int32(0))
    dilation = (np.int32(1), np.int32(1))
    groups = np.int32(1)
    bias = True
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.rand(1, 1, 10, 10).astype(dtype)

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

    # Input 8
    in_channels = np.int32(3)
    out_channels = np.int32(24)
    kernel_size = (np.int32(4), np.int32(4))
    stride = (np.int32(2), np.int32(2))
    padding = (np.int32(1), np.int32(1))
    dilation = (np.int32(1), np.int32(1))
    groups = np.int32(3)
    bias = False
    padding_mode = 'reflect'
    dtype = np.float32
    input = np.random.rand(1, 3, 64, 64).astype(dtype)

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
    in_channels = np.int32(2)
    out_channels = np.int32(4)
    kernel_size = (np.int32(3), np.int32(3))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(1), np.int32(1))
    dilation = (np.int32(2), np.int32(2))
    groups = np.int32(2)
    bias = True
    padding_mode = 'circular'
    dtype = np.float32
    input = np.random.rand(1, 2, 16, 16).astype(dtype)

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

    # Input 10
    in_channels = np.int32(8)
    out_channels = np.int32(8)
    kernel_size = (np.int32(5), np.int32(5))
    stride = (np.int32(2), np.int32(2))
    padding = (np.int32(2), np.int32(2))
    dilation = (np.int32(1), np.int32(1))
    groups = np.int32(8)
    bias = False
    padding_mode = 'replicate'
    dtype = np.float32
    input = np.random.rand(1, 8, 32, 32).astype(dtype)

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
    
    # Input 11
    in_channels = np.int32(4)
    out_channels = np.int32(8)
    kernel_size = (np.int32(3), np.int32(3))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(1), np.int32(1))
    dilation = (np.int32(1), np.int32(1))
    groups = np.int32(1)
    bias = True
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.rand(1, 4, 20, 20).astype(dtype)

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
    
    # Input 12
    in_channels = np.int32(2)
    out_channels = np.int32(2)
    kernel_size = (np.int32(2), np.int32(2))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(0), np.int32(0))
    dilation = (np.int32(1), np.int32(1))
    groups = np.int32(2)
    bias = False
    padding_mode = 'zeros'  # Changed from 'valid'
    dtype = np.float64
    input = np.random.rand(1, 2, 10, 10).astype(dtype)

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
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Conv2d_2"] = conv2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Conv2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Conv2d_2'.")

check_valid('torch.nn.Conv2d', generated_inputs['torch.nn.Conv2d_2'], lib="torch", suffix=2)
