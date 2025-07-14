
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv3d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(2, 3, 10, 10, 10).astype(np.float32)
    input_dict1 = {
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(1, 16, 20, 20, 20).astype(np.float64)
    input_dict2 = {
        "in_channels": 16,
        "out_channels": 32,
        "kernel_size": (3, 5, 2),
        "stride": (2, 1, 1),
        "padding": (4, 2, 0),
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": 'zeros',
        "dtype": torch.float64,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(4, 8, 15, 15, 15).astype(np.float32)
    input_dict3 = {
        "in_channels": 8,
        "out_channels": 16,
        "kernel_size": 5,
        "stride": 2,
        "padding": 1,
        "dilation": 2,
        "groups": 2,
        "bias": True,
        "padding_mode": 'reflect',
        "dtype": torch.float32,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(1, 4, 12, 12, 12).astype(np.float64)
    input_dict4 = {
        "in_channels": 4,
        "out_channels": 12,
        "kernel_size": (2, 3, 4),
        "stride": (1, 2, 1),
        "padding": (1, 0, 1),
        "dilation": (2, 1, 1),
        "groups": 4,
        "bias": False,
        "padding_mode": 'replicate',
        "dtype": torch.float64,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(2, 3, 8, 8, 8).astype(np.float32)
    input_dict5 = {
        "in_channels": 3,
        "out_channels": 6,
        "kernel_size": 2,
        "stride": 1,
        "padding": 'same',
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = np.random.rand(1, 1, 16, 16, 16).astype(np.float64)
    input_dict6 = {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 7,
        "stride": 1,
        "padding": 'valid',
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": 'zeros',
        "dtype": torch.float64,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: depthwise convolution
    input7 = np.random.rand(1, 3, 10, 10, 10).astype(np.float32)
    input_dict7 = {
        "in_channels": 3,
        "out_channels": 6,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 3,
        "bias": True,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(2, 4, 12, 12, 12).astype(np.float64)
    input_dict8 = {
        "in_channels": 4,
        "out_channels": 8,
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "groups": 1,
        "bias": True,
        "padding_mode": 'circular',
        "dtype": torch.float64,
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9
    input9 = np.random.rand(1, 2, 15, 15, 15).astype(np.float32)
    input_dict9 = {
        "in_channels": 2,
        "out_channels": 4,
        "kernel_size": 3,
        "stride": 1,
        "padding": 2,
        "dilation": 1,
        "groups": 1,
        "bias": False,
        "padding_mode": 'reflect',
        "dtype": torch.float32,
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    # Input 10
    input10 = np.random.rand(3, 1, 20, 20, 20).astype(np.float64)
    input_dict10 = {
        "in_channels": 1,
        "out_channels": 2,
        "kernel_size": (5, 5, 5),
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": (2, 2, 2),
        "groups": 1,
        "bias": True,
        "padding_mode": 'replicate',
        "dtype": torch.float64,
        "input": input10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Conv3d"] = conv3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Conv3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Conv3d'.")

check_valid('torch.nn.Conv3d', generated_inputs['torch.nn.Conv3d'], lib="torch", suffix=0)
