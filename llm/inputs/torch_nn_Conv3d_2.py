
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def conv3d_inputs():
    list_of_inputs = []

    # Input 1
    in_channels = np.int64(3)
    out_channels = np.int64(5)
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    dilation = (1, 1, 1)
    groups = np.int64(1)
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float32
    input = torch.randn(1, 3, 10, 10, 10, dtype=torch.float32).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    in_channels = np.int64(3)
    out_channels = np.int64(6)
    kernel_size = (5, 5, 5)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    groups = np.int64(3)
    bias = False
    padding_mode = 'reflect'
    dtype = torch.float64
    input = torch.randn(2, 3, 20, 20, 20, dtype=torch.float64).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    in_channels = np.int64(4)
    out_channels = np.int64(8)
    kernel_size = (2, 3, 4)
    stride = (1, 2, 3)
    padding = (2, 0, 1)
    dilation = (2, 1, 1)
    groups = np.int64(2)
    bias = True
    padding_mode = 'replicate'
    dtype = torch.float16
    input = torch.randn(4, 4, 15, 25, 35, dtype=torch.float16).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    in_channels = np.int64(1)
    out_channels = np.int64(1)
    kernel_size = (1, 1, 1)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    dilation = (1, 1, 1)
    groups = np.int64(1)
    bias = False
    padding_mode = 'circular'
    dtype = torch.float32
    input = torch.randn(1, 1, 5, 5, 5, dtype=torch.float32).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    in_channels = np.int64(2)
    out_channels = np.int64(4)
    kernel_size = (3, 4, 5)
    stride = (1, 1, 1)
    padding = (1, 2, 0)
    dilation = (1, 1, 1)
    groups = np.int64(1)
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float64
    input = torch.randn(1, 2, 12, 15, 18, dtype=torch.float64).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    in_channels = np.int64(5)
    out_channels = np.int64(5)
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    groups = np.int64(5)
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float32
    input = torch.randn(1, 5, 10, 10, 10, dtype=torch.float32).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    in_channels = np.int64(16)
    out_channels = np.int64(32)
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    groups = np.int64(1)
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float32
    input = torch.randn(1, 16, 16, 16, 16, dtype=torch.float32).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    in_channels = np.int64(8)
    out_channels = np.int64(16)
    kernel_size = (5, 5, 5)
    stride = (1, 1, 1)
    padding = (2, 2, 2)
    dilation = (2, 2, 2)
    groups = np.int64(1)
    bias = False
    padding_mode = 'zeros'
    dtype = torch.float32
    input = torch.randn(1, 8, 32, 32, 32, dtype=torch.float32).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    in_channels = np.int64(4)
    out_channels = np.int64(8)
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    groups = np.int64(2)
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float32
    input = torch.randn(1, 4, 10, 10, 10, dtype=torch.float32).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    in_channels = np.int64(3)
    out_channels = np.int64(5)
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    dilation = (1, 1, 1)
    groups = np.int64(1)
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float64
    input = torch.randn(1, 3, 10, 10, 10, dtype=torch.float64).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: depthwise convolution
    in_channels = np.int64(3)
    out_channels = np.int64(3 * 2)  # K = 2
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    groups = np.int64(3)
    bias = True
    padding_mode = 'zeros'
    dtype = torch.float32
    input = torch.randn(1, 3, 10, 10, 10, dtype=torch.float32).numpy()

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
        "dtype": dtype,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Conv3d_2"] = conv3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Conv3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Conv3d_2'.")

check_valid('torch.nn.Conv3d', generated_inputs['torch.nn.Conv3d_2'], lib="torch", suffix=2)
