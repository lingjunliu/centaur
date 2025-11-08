
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def batchnorm3d_inputs():
    list_of_inputs = []

    x = torch.randn(2, 4, 5, 6, 7, dtype=torch.float32).numpy()
    input_dict = {
        "num_features": 4,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(3, 8, 4, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "num_features": 8,
        "eps": 1e-3,
        "momentum": 0.05,
        "affine": True,
        "track_running_stats": False,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(5, 2, 3, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "num_features": 2,
        "eps": 1e-4,
        "momentum": 0.0,
        "affine": False,
        "track_running_stats": True,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.zeros(1, 1, 1, 1, 2, dtype=torch.float32).numpy()
    input_dict = {
        "num_features": 1,
        "eps": 1e-6,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": True,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (torch.rand(2, 3, 2, 3, 4, dtype=torch.float32) * 200 - 100).numpy()
    input_dict = {
        "num_features": 3,
        "eps": 1e-2,
        "momentum": 1.0,
        "affine": False,
        "track_running_stats": False,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(4, 16, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "num_features": 16,
        "eps": 1e-5,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": True,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(7, 5, 1, 3, 1, dtype=torch.float32).numpy()
    input_dict = {
        "num_features": 5,
        "eps": 1e-5,
        "momentum": 0.7,
        "affine": True,
        "track_running_stats": False,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(2, 6, 3, 1, 5, dtype=torch.float32).numpy()
    input_dict = {
        "num_features": 6,
        "eps": 1e-7,
        "momentum": 0.3,
        "affine": False,
        "track_running_stats": True,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (torch.randn(1, 10, 7, 7, 7, dtype=torch.float32) * 100).numpy()
    input_dict = {
        "num_features": 10,
        "eps": 5e-5,
        "momentum": 0.15,
        "affine": True,
        "track_running_stats": True,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(9, 12, 2, 5, 3, dtype=torch.float32).numpy()
    input_dict = {
        "num_features": 12,
        "eps": 1e-5,
        "momentum": 0.5,
        "affine": False,
        "track_running_stats": False,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(2, 32, 1, 1, 10, dtype=torch.float32).numpy()
    input_dict = {
        "num_features": 32,
        "eps": 1e-5,
        "momentum": 0.01,
        "affine": True,
        "track_running_stats": True,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = torch.randn(3, 4, 9, 9, 2, dtype=torch.float32).numpy()
    input_dict = {
        "num_features": 4,
        "eps": 1e-1,
        "momentum": 0.4,
        "affine": True,
        "track_running_stats": True,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.BatchNorm3d"] = batchnorm3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BatchNorm3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BatchNorm3d'.")


check_valid('torch.nn.BatchNorm3d', generated_inputs['torch.nn.BatchNorm3d'], lib="torch", suffix=0)
