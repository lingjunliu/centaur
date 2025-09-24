
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def lazy_instance_norm2d_inputs():
    list_of_inputs = []

    # Input 1: Basic input (N, C, H, W)
    input1 = torch.randn(2, 3, 32, 32).numpy()
    input_dict1 = {
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Input (C, H, W)
    input2 = torch.randn(3, 32, 32).numpy()
    input_dict2 = {
        "eps": 1e-4,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different eps and momentum
    input3 = torch.randn(4, 5, 16, 16).numpy()
    input_dict3 = {
        "eps": 1e-3,
        "momentum": 0.3,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different dtype (torch.float64)
    input4 = torch.randn(1, 2, 64, 64, dtype=torch.float64).numpy()
    input_dict4 = {
        "eps": 1e-6,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float64,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Different input size and smaller values
    input5 = torch.randn(2, 1, 8, 8).numpy() * 0.1
    input_dict5 = {
        "eps": 1e-7,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LazyInstanceNorm2d"] = lazy_instance_norm2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.LazyInstanceNorm2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LazyInstanceNorm2d'.")

check_valid('torch.nn.LazyInstanceNorm2d', generated_inputs['torch.nn.LazyInstanceNorm2d'], lib="torch")
