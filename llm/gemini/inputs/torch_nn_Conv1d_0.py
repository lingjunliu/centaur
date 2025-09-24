
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv1d_inputs():
    list_of_inputs = []

    # Input 1
    in_channels = np.int32(3)
    out_channels = np.int32(5)
    kernel_size = np.int32(2)
    stride = np.int32(1)
    padding = np.int32(0)
    dilation = np.int32(1)
    groups = np.int32(1)
    bias = np.bool_(True)
    padding_mode = "zeros"
    dtype = torch.float32
    input = torch.randn(1, 3, 10, dtype=torch.float32).numpy()

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
    in_channels = np.int32(5)
    out_channels = np.int32(10)
    kernel_size = np.int32(3)
    stride = np.int32(2)
    padding = np.int32(1)
    dilation = np.int32(1)
    groups = np.int32(1)
    bias = np.bool_(False)
    padding_mode = "reflect"
    dtype = torch.float64
    input = torch.randn(2, 5, 20, dtype=torch.float64).numpy()

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
    in_channels = np.int32(4)
    out_channels = np.int32(8)
    kernel_size = np.int32(4)
    stride = np.int32(1)
    padding = np.int32(2)
    dilation = np.int32(2)
    groups = np.int32(2)
    bias = np.bool_(True)
    padding_mode = "replicate"
    dtype = torch.float32
    input = torch.randn(1, 4, 30, dtype=torch.float32).numpy()

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
    in_channels = np.int32(2)
    out_channels = np.int32(4)
    kernel_size = np.int32(5)
    stride = np.int32(3)
    padding = np.int32(0)
    dilation = np.int32(1)
    groups = np.int32(1)
    bias = np.bool_(False)
    padding_mode = "circular"
    dtype = torch.float64
    input = torch.randn(3, 2, 40, dtype=torch.float64).numpy()

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
    in_channels = np.int32(1)
    out_channels = np.int32(1)
    kernel_size = np.int32(1)
    stride = np.int32(1)
    padding = np.int32(0)
    dilation = np.int32(1)
    groups = np.int32(1)
    bias = np.bool_(True)
    padding_mode = "zeros"
    dtype = torch.float32
    input = torch.randn(1, 1, 50, dtype=torch.float32).numpy()

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
    in_channels = np.int32(7)
    out_channels = np.int32(14)
    kernel_size = np.int32(3)
    stride = np.int32(2)
    padding = np.int32(1)
    dilation = np.int32(1)
    groups = np.int32(7)
    bias = np.bool_(False)
    padding_mode = "reflect"
    dtype = torch.float64
    input = torch.randn(2, 7, 25, dtype=torch.float64).numpy()

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
    in_channels = np.int32(6)
    out_channels = np.int32(12)
    kernel_size = np.int32(2)
    stride = np.int32(1)
    padding = np.int32(2)
    dilation = np.int32(2)
    groups = np.int32(6)
    bias = np.bool_(True)
    padding_mode = "replicate"
    dtype = torch.float32
    input = torch.randn(1, 6, 35, dtype=torch.float32).numpy()

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
    in_channels = np.int32(8)
    out_channels = np.int32(8)
    kernel_size = np.int32(4)
    stride = np.int32(3)
    padding = np.int32(0)
    dilation = np.int32(1)
    groups = np.int32(8)
    bias = np.bool_(False)
    padding_mode = "circular"
    dtype = torch.float64
    input = torch.randn(3, 8, 45, dtype=torch.float64).numpy()

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
    in_channels = np.int32(9)
    out_channels = np.int32(18)
    kernel_size = np.int32(1)
    stride = np.int32(1)
    padding = np.int32(0)
    dilation = np.int32(1)
    groups = np.int32(9)
    bias = np.bool_(True)
    padding_mode = "zeros"
    dtype = torch.float32
    input = torch.randn(1, 9, 55, dtype=torch.float32).numpy()

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
    in_channels = np.int32(10)
    out_channels = np.int32(15)
    kernel_size = np.int32(3)
    stride = np.int32(2)
    padding = np.int32(1)
    dilation = np.int32(1)
    groups = np.int32(5)
    bias = np.bool_(False)
    padding_mode = "reflect"
    dtype = torch.float64
    input = torch.randn(2, 10, 28, dtype=torch.float64).numpy()

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
generated_inputs["torch.nn.Conv1d"] = conv1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Conv1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Conv1d'.")

check_valid('torch.nn.Conv1d', generated_inputs['torch.nn.Conv1d'], lib="torch", suffix=0)
