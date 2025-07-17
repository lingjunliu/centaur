
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
    kernel_size = np.int32(3)
    stride = np.int32(1)
    padding = np.int32(1)
    output_padding = np.int32(0)
    groups = np.int32(1)
    bias = np.bool_(True)
    dilation = np.int32(1)
    padding_mode = "zeros"
    dtype = np.float32
    input = torch.randn(1, 3, 10, 10).numpy()
    output_size = None

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
    in_channels = np.int32(16)
    out_channels = np.int32(32)
    kernel_size = (np.int32(5), np.int32(3))
    stride = (np.int32(2), np.int32(1))
    padding = (np.int32(2), np.int32(1))
    output_padding = (np.int32(1), np.int32(0))
    groups = np.int32(1)
    bias = np.bool_(False)
    dilation = (np.int32(1), np.int32(1))
    padding_mode = "zeros"
    dtype = np.float64
    input = torch.randn(4, 16, 20, 30).numpy()
    output_size = None

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
    in_channels = np.int32(1)
    out_channels = np.int32(1)
    kernel_size = np.int32(7)
    stride = np.int32(3)
    padding = np.int32(3)
    output_padding = np.int32(2)
    groups = np.int32(1)
    bias = np.bool_(True)
    dilation = np.int32(1)
    padding_mode = "zeros"
    dtype = np.float32
    input = torch.randn(1, 1, 50, 50).numpy()
    output_size = None

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
generated_inputs["torch.nn.ConvTranspose2d"] = convtranspose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConvTranspose2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConvTranspose2d'.")

check_valid('torch.nn.ConvTranspose2d', generated_inputs['torch.nn.ConvTranspose2d'], lib="torch", suffix=0)
