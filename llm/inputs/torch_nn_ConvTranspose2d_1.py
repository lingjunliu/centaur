
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def convtranspose2d_inputs():
    list_of_inputs = []

    # Input 1
    in_channels = 3
    out_channels = 6
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    output_padding = (0, 0)
    groups = 1
    bias = True
    dilation = (1, 1)
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.randn(1, in_channels, 24, 24).astype(dtype)
    output_size = (26,26)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "bias": bias,
        "dilation": dilation,
        "padding_mode": padding_mode,
        "dtype": np.dtype(dtype),
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    in_channels = 3
    out_channels = 6
    kernel_size = (3, 5)
    stride = (2, 1)
    padding = (1, 2)
    output_padding = (1, 0)
    groups = 1
    bias = False
    dilation = (1, 1)
    padding_mode = 'zeros'
    dtype = np.float64
    input = np.random.randn(1, in_channels, 12, 24).astype(dtype)
    output_size = (23, 24)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "bias": bias,
        "dilation": dilation,
        "padding_mode": padding_mode,
        "dtype": np.dtype(dtype),
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    in_channels = 4
    out_channels = 8
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = (0, 0)
    output_padding = (0, 0)
    groups = 2
    bias = True
    dilation = (1, 1)
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.randn(1, in_channels, 16, 16).astype(dtype)
    output_size = (32, 32)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "bias": bias,
        "dilation": dilation,
        "padding_mode": padding_mode,
        "dtype": np.dtype(dtype),
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    in_channels = 2
    out_channels = 4
    kernel_size = (4, 4)
    stride = (3, 3)
    padding = (1, 1)
    output_padding = (0, 0)
    groups = 1
    bias = False
    dilation = (2, 2)
    padding_mode = 'zeros'
    dtype = np.float64
    input = np.random.randn(1, in_channels, 8, 8).astype(dtype)
    output_size = (26, 26) # changed output size

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "bias": bias,
        "dilation": dilation,
        "padding_mode": padding_mode,
        "dtype": np.dtype(dtype),
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    in_channels = 1
    out_channels = 1
    kernel_size = (1, 1)
    stride = (1, 1)
    padding = (0, 0)
    output_padding = (0, 0)
    groups = 1
    bias = True
    dilation = (1, 1)
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.randn(1, in_channels, 32, 32).astype(dtype)
    output_size = (32, 32)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "bias": bias,
        "dilation": dilation,
        "padding_mode": padding_mode,
        "dtype": np.dtype(dtype),
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    in_channels = 8
    out_channels = 4
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (0, 0)
    groups = 1
    bias = True
    dilation = (1, 1)
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.randn(1, in_channels, 16, 16).astype(dtype)
    output_size = (16, 16)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "bias": bias,
        "dilation": dilation,
        "padding_mode": padding_mode,
        "dtype": np.dtype(dtype),
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    output_padding = (1, 1)
    groups = 1
    bias = True
    dilation = (1, 1)
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.randn(1, in_channels, 16, 16).astype(dtype)
    output_size = (31, 31)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "bias": bias,
        "dilation": dilation,
        "padding_mode": padding_mode,
        "dtype": np.dtype(dtype),
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    in_channels = 3
    out_channels = 3
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    output_padding = (0, 0)
    groups = 3
    bias = False
    dilation = (1, 1)
    padding_mode = 'zeros'
    dtype = np.float64
    input = np.random.randn(1, in_channels, 5, 5).astype(dtype)
    output_size = (6, 6)
    
    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "bias": bias,
        "dilation": dilation,
        "padding_mode": padding_mode,
        "dtype": np.dtype(dtype),
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    in_channels = 16
    out_channels = 8
    kernel_size = (4, 4)
    stride = (2, 2)
    padding = (1, 1)
    output_padding = (0, 0)
    groups = 1
    bias = True
    dilation = (1, 1)
    padding_mode = 'zeros'
    dtype = np.float32
    input = np.random.randn(1, in_channels, 32, 32).astype(dtype)
    output_size = (62, 62)

    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "bias": bias,
        "dilation": dilation,
        "padding_mode": padding_mode,
        "dtype": np.dtype(dtype),
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    output_padding = (1, 1)
    groups = 4
    bias = False
    dilation = (1, 1)
    padding_mode = 'zeros'
    dtype = np.float64
    input = np.random.randn(1, in_channels, 16, 16).astype(dtype)
    output_size = (31, 31)
    
    input_dict = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "bias": bias,
        "dilation": dilation,
        "padding_mode": padding_mode,
        "dtype": np.dtype(dtype),
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ConvTranspose2d_1"] = convtranspose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConvTranspose2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConvTranspose2d_1'.")

check_valid('torch.nn.ConvTranspose2d', generated_inputs['torch.nn.ConvTranspose2d_1'], lib="torch", suffix=1)
