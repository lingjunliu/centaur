
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def convtranspose2d_inputs():
    list_of_inputs = []

    # Input 1
    in_channels = np.int32(3)
    out_channels = np.int32(16)
    kernel_size = (np.int32(3), np.int32(5))
    stride = (np.int32(2), np.int32(1))
    padding = (np.int32(1), np.int32(2))
    output_padding = (np.int32(0), np.int32(0))
    groups = np.int32(1)
    bias = True
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(2, 3, 10, 10).astype(np.float32)
    output_size = (np.int32(2), np.int32(16), np.int32(20), np.int32(10))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    in_channels = np.int32(1)
    out_channels = np.int32(1)
    kernel_size = (np.int32(4), np.int32(4))
    stride = (np.int32(2), np.int32(2))
    padding = (np.int32(1), np.int32(1))
    output_padding = (np.int32(1), np.int32(1))
    groups = np.int32(1)
    bias = False
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float64
    input = np.random.randn(1, 1, 5, 5).astype(np.float64)
    output_size = (np.int32(1), np.int32(1), np.int32(10), np.int32(10))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    in_channels = np.int32(32)
    out_channels = np.int32(64)
    kernel_size = (np.int32(2), np.int32(2))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(0), np.int32(0))
    output_padding = (np.int32(0), np.int32(0))
    groups = np.int32(1)
    bias = True
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(4, 32, 20, 30).astype(np.float32)
    output_size = (np.int32(4), np.int32(64), np.int32(21), np.int32(31))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    in_channels = np.int32(4)
    out_channels = np.int32(8)
    kernel_size = (np.int32(3), np.int32(3))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(1), np.int32(1))
    output_padding = (np.int32(0), np.int32(0))
    groups = np.int32(2)
    bias = True
    dilation = (np.int32(2), np.int32(2))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(1, 4, 16, 16).astype(np.float32)
    output_size = (np.int32(1), np.int32(8), np.int32(28), np.int32(28))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    in_channels = np.int32(1)
    out_channels = np.int32(1)
    kernel_size = (np.int32(3), np.int32(3))
    stride = (np.int32(2), np.int32(2))
    padding = (np.int32(1), np.int32(1))
    output_padding = (np.int32(1), np.int32(1))
    groups = np.int32(1)
    bias = True
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(1, 1, 6, 6).astype(np.float32)
    output_size = (np.int32(1), np.int32(1), np.int32(12), np.int32(12))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    in_channels = np.int32(3)
    out_channels = np.int32(3)
    kernel_size = (np.int32(3), np.int32(3))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(0), np.int32(0))
    output_padding = (np.int32(0), np.int32(0))
    groups = np.int32(3)
    bias = False
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(1, 3, 10, 10).astype(np.float32)
    output_size = (np.int32(1), np.int32(3), np.int32(12), np.int32(12))
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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    in_channels = np.int32(16)
    out_channels = np.int32(33)
    kernel_size = (np.int32(3), np.int32(5))
    stride = (np.int32(2), np.int32(1))
    padding = (np.int32(4), np.int32(2))
    output_padding = (np.int32(0), np.int32(0))
    groups = np.int32(1)
    bias = True
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(20, 16, 50, 100).astype(np.float32)
    output_size = (np.int32(20), np.int32(33), np.int32(95), np.int32(99))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    in_channels = np.int32(16)
    out_channels = np.int32(16)
    kernel_size = (np.int32(3), np.int32(3))
    stride = (np.int32(2), np.int32(2))
    padding = (np.int32(1), np.int32(1))
    output_padding = (np.int32(0), np.int32(0))
    groups = np.int32(1)
    bias = True
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(1, 16, 6, 6).astype(np.float32)
    output_size = (np.int32(1), np.int32(16), np.int32(11), np.int32(11))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    in_channels = np.int32(2)
    out_channels = np.int32(4)
    kernel_size = (np.int32(2), np.int32(2))
    stride = (np.int32(2), np.int32(2))
    padding = (np.int32(0), np.int32(0))
    output_padding = (np.int32(0), np.int32(0))
    groups = np.int32(1)
    bias = True
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(1, 2, 4, 4).astype(np.float32)
    output_size = (np.int32(1), np.int32(4), np.int32(8), np.int32(8))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    in_channels = np.int32(1)
    out_channels = np.int32(1)
    kernel_size = (np.int32(2), np.int32(2))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(0), np.int32(0))
    output_padding = (np.int32(0), np.int32(0))
    groups = np.int32(1)
    bias = True
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(1, 1, 3, 3).astype(np.float32)
    output_size = (np.int32(1), np.int32(1), np.int32(4), np.int32(4))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    in_channels = np.int32(2)
    out_channels = np.int32(2)
    kernel_size = (np.int32(3), np.int32(3))
    stride = (np.int32(2), np.int32(2))
    padding = (np.int32(1), np.int32(1))
    output_padding = (np.int32(1), np.int32(1))
    groups = np.int32(2)
    bias = False
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(1, 2, 4, 4).astype(np.float32)
    output_size = (np.int32(1), np.int32(2), np.int32(8), np.int32(8))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    in_channels = np.int32(1)
    out_channels = np.int32(1)
    kernel_size = (np.int32(2), np.int32(2))
    stride = (np.int32(2), np.int32(2))
    padding = (np.int32(0), np.int32(0))
    output_padding = (np.int32(0), np.int32(0))
    groups = np.int32(1)
    bias = True
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(1, 1, 5, 5).astype(np.float32)
    output_size = (np.int32(1), np.int32(1), np.int32(8), np.int32(8))

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
        "dtype": dtype,
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    in_channels = np.int32(4)
    out_channels = np.int32(8)
    kernel_size = (np.int32(2), np.int32(2))
    stride = (np.int32(1), np.int32(1))
    padding = (np.int32(0), np.int32(0))
    output_padding = (np.int32(0), np.int32(0))
    groups = np.int32(1)
    bias = True
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = torch.float32
    input = np.random.randn(1, 4, 16, 16).astype(np.float32)
    output_size = (np.int32(1), np.int32(8), np.int32(17), np.int32(17))

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
        "dtype": dtype,
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
