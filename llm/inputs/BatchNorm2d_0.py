
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def BatchNorm2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 100, 35, 45).numpy()
    input_dict1 = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(5, 64, 28, 28).numpy()
    input_dict2 = {
        "num_features": 64,
        "eps": 1e-04,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(10, 32, 14, 14).numpy()
    input_dict3 = {
        "num_features": 32,
        "eps": 1e-06,
        "momentum": 0.05,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 128, 56, 56).numpy()
    input_dict4 = {
        "num_features": 128,
        "eps": 1e-03,
        "momentum": 0.3,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(8, 16, 7, 7).numpy()
    input_dict5 = {
        "num_features": 16,
        "eps": 1e-07,
        "momentum": 0.01,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.BatchNorm2d"] = BatchNorm2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.BatchNorm2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BatchNorm2d'.")

check_valid('torch.nn.BatchNorm2d', generated_inputs['torch.nn.BatchNorm2d'], lib="torch")
