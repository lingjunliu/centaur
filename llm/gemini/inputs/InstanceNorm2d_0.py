
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def InstanceNorm2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with N, C, H, W
    input1 = np.random.randn(2, 3, 32, 32).astype(np.float32)
    input_dict1 = {
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: With affine=True
    input2 = np.random.randn(1, 5, 16, 16).astype(np.float32)
    input_dict2 = {
        "num_features": 5,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": False,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: With track_running_stats=True
    input3 = np.random.randn(4, 7, 64, 64).astype(np.float32)
    input_dict3 = {
        "num_features": 7,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": True,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different momentum and eps
    input4 = np.random.randn(1, 10, 8, 8).astype(np.float32)
    input_dict4 = {
        "num_features": 10,
        "eps": 1e-3,
        "momentum": 0.5,
        "affine": False,
        "track_running_stats": False,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Input with C, H, W
    input5 = np.random.randn(12, 24, 24).astype(np.float32)
    input_dict5 = {
        "num_features": 12,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.InstanceNorm2d"] = InstanceNorm2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.InstanceNorm2d', generated_inputs['torch.nn.InstanceNorm2d'], lib="torch")
