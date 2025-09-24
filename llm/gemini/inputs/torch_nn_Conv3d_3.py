
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
    dtype1 = np.dtype(np.float32)
    input_dict1 = {
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": 3,
        "stride": 1,
        "padding": "valid",
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "padding_mode": "zeros",
        "dtype": dtype1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(1, 4, 8, 12, 16).astype(np.float64)
    dtype2 = np.dtype(np.float64)

    input_dict2 = {
        "in_channels": 4,
        "out_channels": 8,
        "kernel_size": 2,
        "stride": 1,
        "padding": "same",
        "dilation": 1,
        "groups": 2,
        "bias": False,
        "padding_mode": "replicate",
        "dtype": dtype2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(3, 2, 5, 7, 9).astype(np.float32)
    dtype3 = np.dtype(np.float32)

    input_dict3 = {
        "in_channels": 2,
        "out_channels": 4,
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "groups": 1,
        "bias": True,
        "padding_mode": "zeros",
        "dtype": dtype3,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Conv3d_3"] = conv3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Conv3d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Conv3d_3'.")

check_valid('torch.nn.Conv3d', generated_inputs['torch.nn.Conv3d_3'], lib="torch", suffix=3)
