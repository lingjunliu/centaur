
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import torch.nn as nn
import numpy as np
import copy

def GroupNorm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float32 and affine=True
    input1 = torch.randn(20, 6, 10, 10).numpy()
    num_groups1 = 3
    num_channels1 = 6
    input_dict1 = {
        "num_groups": num_groups1,
        "num_channels": num_channels1,
        "eps": 1e-05,
        "affine": True,
        "input": input1,
        "dtype": torch.float32  # Use torch.dtype here
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different number of groups, equivalent to InstanceNorm
    input2 = torch.randn(5, 8, 7, 7).numpy()
    num_groups2 = 8
    num_channels2 = 8
    input_dict2 = {
        "num_groups": num_groups2,
        "num_channels": num_channels2,
        "eps": 1e-08,
        "affine": False,
        "input": input2,
        "dtype": torch.float32 # Use torch.dtype here
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Equivalent to LayerNorm
    input3 = torch.randn(10, 4, 5, 5).numpy()
    num_groups3 = 1
    num_channels3 = 4
    input_dict3 = {
        "num_groups": num_groups3,
        "num_channels": num_channels3,
        "eps": 1e-04,
        "affine": True,
        "input": input3,
        "dtype": torch.float32 # Use torch.dtype here
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 7: Different epsilon
    input7 = torch.randn(1, 32, 8, 8).numpy()
    num_groups7 = 4
    num_channels7 = 32
    input_dict7 = {
        "num_groups": num_groups7,
        "num_channels": num_channels7,
        "eps": 1e-02,
        "affine": True,
        "input": input7,
        "dtype": torch.float32 # Use torch.dtype here
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.GroupNorm"] = GroupNorm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.GroupNorm', generated_inputs['torch.nn.GroupNorm'], lib="torch")
