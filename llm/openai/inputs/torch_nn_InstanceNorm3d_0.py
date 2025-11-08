
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def instance_norm3d_inputs():
    list_of_inputs = []

    arr = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32).numpy()
    list_of_inputs.append({
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": arr
    })

    arr = torch.randn(4, 3, 3, 3, dtype=torch.float32).numpy()
    list_of_inputs.append({
        "num_features": 4,
        "eps": 1e-5,
        "momentum": 0.0,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": arr
    })

    arr = torch.randn(5, 1, 8, 8, 8, dtype=torch.float64).numpy()
    list_of_inputs.append({
        "num_features": 1,
        "eps": 1e-3,
        "momentum": 0.9,
        "affine": False,
        "track_running_stats": True,
        "dtype": torch.float64,
        "input": arr
    })

    arr = torch.randn(1, 2, 3, 4, dtype=torch.float32).numpy()
    list_of_inputs.append({
        "num_features": 1,
        "eps": 1e-4,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": arr
    })

    arr = torch.randn(3, 2, 5, 7, 9, dtype=torch.float32).numpy()
    list_of_inputs.append({
        "num_features": 2,
        "eps": 1e-5,
        "momentum": 0.7,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": arr
    })

    arr = torch.randn(2, 16, 4, 4, 4, dtype=torch.float32).numpy()
    list_of_inputs.append({
        "num_features": 16,
        "eps": 1e-8,
        "momentum": 0.3,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": arr
    })

    arr = torch.randn(2, 2, 6, 5, 4, dtype=torch.float64).numpy()
    list_of_inputs.append({
        "num_features": 2,
        "eps": 1e-6,
        "momentum": 0.5,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float64,
        "input": arr
    })

    arr = torch.randn(2, 10, 4, 4, dtype=torch.float64).numpy()
    list_of_inputs.append({
        "num_features": 2,
        "eps": 1e-5,
        "momentum": 1.0,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float64,
        "input": arr
    })

    arr = torch.randn(1, 3, 2, 2, 2, dtype=torch.float64).numpy()
    list_of_inputs.append({
        "num_features": 3,
        "eps": 1e-12,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": False,
        "dtype": torch.float64,
        "input": arr
    })

    arr = torch.randn(4, 5, 3, 2, 1, dtype=torch.float32).numpy()
    list_of_inputs.append({
        "num_features": 5,
        "eps": 5e-6,
        "momentum": 0.4,
        "affine": False,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": arr
    })

    return list_of_inputs

generated_inputs["torch.nn.InstanceNorm3d"] = instance_norm3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.InstanceNorm3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.InstanceNorm3d'.")


check_valid('torch.nn.InstanceNorm3d', generated_inputs['torch.nn.InstanceNorm3d'], lib="torch", suffix=0)
