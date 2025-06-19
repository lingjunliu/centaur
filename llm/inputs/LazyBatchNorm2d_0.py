
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def lazy_batchnorm2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 224, 224).numpy()
    input_dict1 = {
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 5, 64, 64).numpy()
    input_dict2 = {
        "eps": 1e-3,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": input2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(4, 10, 128, 128).numpy()
    input_dict3 = {
        "eps": 1e-8,
        "momentum": None,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 32, 32).numpy()
    input_dict4 = {
        "eps": 1e-2,
        "momentum": 0.5,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": input4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 8, 256, 256).numpy()
    input_dict5 = {
        "eps": 1e-7,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": input5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LazyBatchNorm2d"] = lazy_batchnorm2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.LazyBatchNorm2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LazyBatchNorm2d'.")

check_valid('torch.nn.LazyBatchNorm2d', generated_inputs['torch.nn.LazyBatchNorm2d'], lib="torch")
