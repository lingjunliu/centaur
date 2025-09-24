
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def lazy_batch_norm1d_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 3: Float tensor with None momentum - MODIFIED to have batch size > 1
    input3 = torch.randn(2, 7).numpy()
    input_dict3 = {
        "eps": 1e-4,
        "momentum": None,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Float tensor with specified dtype
    input4 = torch.randn(3, 2).numpy()
    input_dict4 = {
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float64,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large float tensor
    input5 = torch.randn(10, 12).numpy()
    input_dict5 = {
        "eps": 1e-6,
        "momentum": 0.3,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Tensor with negative values
    input6 = torch.randn(2, 4) * -1.0
    input6 = input6.numpy()
    input_dict6 = {
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Float16 tensor
    input7 = torch.randn(2, 5, dtype=torch.float16).numpy()
    input_dict7 = {
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float16,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LazyBatchNorm1d"] = lazy_batch_norm1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.LazyBatchNorm1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LazyBatchNorm1d'.")

check_valid('torch.nn.LazyBatchNorm1d', generated_inputs['torch.nn.LazyBatchNorm1d'], lib="torch")
