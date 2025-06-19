
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def InstanceNorm3d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 100, 35, 45, 10).numpy()
    input_dict1 = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": input1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict2 = {
        "num_features": 3,
        "eps": 1e-04,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 64, 16, 16, 16).numpy()
    input_dict3 = {
        "num_features": 64,
        "eps": 1e-08,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(5, 128, 8, 8, 8).numpy()
    input_dict4 = {
        "num_features": 128,
        "eps": 1e-03,
        "momentum": 0.3,
        "affine": True,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": input4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(10, 32, 24, 24, 24).numpy()
    input_dict5 = {
        "num_features": 32,
        "eps": 1e-06,
        "momentum": 0.15,
        "affine": False,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.InstanceNorm3d"] = InstanceNorm3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.InstanceNorm3d', generated_inputs['torch.nn.InstanceNorm3d'], lib="torch")
