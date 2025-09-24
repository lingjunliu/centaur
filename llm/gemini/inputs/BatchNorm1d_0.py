
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def BatchNorm1d_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D input
    input1 = np.random.randn(20, 100).astype(np.float32)
    input_dict1 = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D input
    input2 = np.random.randn(20, 100, 50).astype(np.float32)
    input_dict2 = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Without affine parameters
    input3 = np.random.randn(20, 100).astype(np.float32)
    input_dict3 = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Without tracking running stats
    input4 = np.random.randn(20, 100).astype(np.float32)
    input_dict4 = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different momentum
    input5 = np.random.randn(20, 100).astype(np.float32)
    input_dict5 = {
        "num_features": 100,
        "eps": 1e-05,
        "momentum": 0.5,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.BatchNorm1d"] = BatchNorm1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.BatchNorm1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BatchNorm1d'.")

check_valid('torch.nn.BatchNorm1d', generated_inputs['torch.nn.BatchNorm1d'], lib="torch")
