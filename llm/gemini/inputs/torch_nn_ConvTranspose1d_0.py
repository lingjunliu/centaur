
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def convtranspose1d_inputs():
    list_of_inputs = []

    # Input 1
    in_channels = 3
    out_channels = 5
    kernel_size = (3,)
    stride = (1,)
    padding = (0,)
    output_padding = (0,)
    groups = 1
    bias = True
    dilation = (1,)
    padding_mode = 'zeros'
    dtype = np.float32
    input_tensor = np.random.rand(2, in_channels, 10).astype(dtype)

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
        "dtype": torch.float32,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    in_channels = 4
    out_channels = 8
    kernel_size = (5,)
    stride = (2,)
    padding = (1,)
    output_padding = (1,)
    groups = 2
    bias = False
    dilation = (2,)
    padding_mode = 'zeros'
    dtype = np.float64
    input_tensor = np.random.rand(1, in_channels, 15).astype(dtype)

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
        "dtype": torch.float64,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3
    in_channels = 2
    out_channels = 4
    kernel_size = (7,)
    stride = (3,)
    padding = (2,)
    output_padding = (0,)
    groups = 1
    bias = True
    dilation = (1,)
    padding_mode = 'zeros'
    dtype = np.float32
    input_tensor = np.random.rand(3, in_channels, 20).astype(dtype)

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
        "dtype": torch.float32,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    in_channels = 6
    out_channels = 6
    kernel_size = (3,)
    stride = (1,)
    padding = (1,)
    output_padding = (0,)
    groups = 3
    bias = False
    dilation = (1,)
    padding_mode = 'zeros'
    dtype = np.float64
    input_tensor = np.random.rand(1, in_channels, 12).astype(dtype)

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
        "dtype": torch.float64,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    in_channels = 1
    out_channels = 1
    kernel_size = (2,)
    stride = (2,)
    padding = (0,)
    output_padding = (1,)
    groups = 1
    bias = True
    dilation = (1,)
    padding_mode = 'zeros'
    dtype = np.float32
    input_tensor = np.random.rand(4, in_channels, 8).astype(dtype)

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
        "dtype": torch.float32,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    in_channels = 7
    out_channels = 14
    kernel_size = (4,)
    stride = (2,)
    padding = (1,)
    output_padding = (0,)
    groups = 7
    bias = True
    dilation = (1,)
    padding_mode = 'zeros'
    dtype = np.float32
    input_tensor = np.random.rand(1, in_channels, 16).astype(dtype)

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
        "dtype": torch.float32,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    in_channels = 3
    out_channels = 1
    kernel_size = (5,)
    stride = (1,)
    padding = (2,)
    output_padding = (0,)
    groups = 1
    bias = False
    dilation = (1,)
    padding_mode = 'zeros'
    dtype = np.float64
    input_tensor = np.random.rand(2, in_channels, 25).astype(dtype)

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
        "dtype": torch.float64,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    in_channels = 5
    out_channels = 5
    kernel_size = (3,)
    stride = (3,)
    padding = (0,)
    output_padding = (2,)
    groups = 5
    bias = True
    dilation = (1,)
    padding_mode = 'zeros'
    dtype = np.float32
    input_tensor = np.random.rand(1, in_channels, 11).astype(dtype)

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
        "dtype": torch.float32,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    in_channels = 4
    out_channels = 12
    kernel_size = (6,)
    stride = (2,)
    padding = (1,)
    output_padding = (1,)
    groups = 2
    bias = False
    dilation = (2,)
    padding_mode = 'zeros'
    dtype = np.float64
    input_tensor = np.random.rand(3, in_channels, 18).astype(dtype)

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
        "dtype": torch.float64,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    in_channels = 2
    out_channels = 1
    kernel_size = (4,)
    stride = (1,)
    padding = (1,)
    output_padding = (0,)
    groups = 1
    bias = True
    dilation = (1,)
    padding_mode = 'zeros'
    dtype = np.float32
    input_tensor = np.random.rand(4, in_channels, 22).astype(dtype)

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
        "dtype": torch.float32,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    in_channels = 2
    out_channels = 2
    kernel_size = (3,)
    stride = (1,)
    padding = (1,)
    output_padding = (0,)
    groups = 2
    bias = False
    dilation = (1,)
    padding_mode = 'zeros'
    dtype = np.float64
    input_tensor = np.random.rand(1, in_channels, 10).astype(dtype)

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
        "dtype": torch.float64,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ConvTranspose1d"] = convtranspose1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConvTranspose1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConvTranspose1d'.")

check_valid('torch.nn.ConvTranspose1d', generated_inputs['torch.nn.ConvTranspose1d'], lib="torch", suffix=0)
