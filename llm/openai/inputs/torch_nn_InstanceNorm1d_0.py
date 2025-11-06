
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def instance_norm1d_inputs():
    list_of_inputs = []

    # 1
    input_arr = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float32'),
        "input": input_arr
    }))

    # 2
    input_arr = np.random.randn(5, 7).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 5,
        "eps": 1e-3,
        "momentum": 0.0,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float64'),
        "input": input_arr
    }))

    # 3
    input_arr = np.random.randn(4, 1, 10).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 1,
        "eps": 1e-6,
        "momentum": 0.5,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float32'),
        "input": input_arr
    }))

    # 4 (ensure L > 1)
    input_arr = np.random.randn(1, 8, 2).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 8,
        "eps": 1e-4,
        "momentum": 0.9,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float32'),
        "input": input_arr
    }))

    # 5
    input_arr = np.random.randn(3, 16, 20).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 16,
        "eps": 1e-5,
        "momentum": 1.0,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float64'),
        "input": input_arr
    }))

    # 6
    input_arr = np.random.randn(2, 2, 50).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 2,
        "eps": 1e-2,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float32'),
        "input": input_arr
    }))

    # 7
    input_arr = np.random.randn(10, 4, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 4,
        "eps": 1e-7,
        "momentum": 0.3,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float32'),
        "input": input_arr
    }))

    # 8
    input_arr = np.random.randn(32, 6).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 32,
        "eps": 5e-5,
        "momentum": 0.75,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float64'),
        "input": input_arr
    }))

    # 9
    input_arr = (np.random.randn(1, 1, 7) * 1000).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 1,
        "eps": 1e-8,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float32'),
        "input": input_arr
    }))

    # 10
    input_arr = np.random.randn(7, 9, 11).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 9,
        "eps": 1e-3,
        "momentum": 0.99,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float32'),
        "input": input_arr
    }))

    # 11
    input_arr = np.random.randn(4, 12).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 4,
        "eps": 1e-5,
        "momentum": 0.4,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float64'),
        "input": input_arr
    }))

    # 12
    input_arr = np.random.randn(6, 64, 3).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_features": 64,
        "eps": 1e-2,
        "momentum": 0.15,
        "affine": False,
        "track_running_stats": False,
        "dtype": np.dtype('float32'),
        "input": input_arr
    }))

    return list_of_inputs

generated_inputs["torch.nn.InstanceNorm1d"] = instance_norm1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.InstanceNorm1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.InstanceNorm1d'.")


check_valid('torch.nn.InstanceNorm1d', generated_inputs['torch.nn.InstanceNorm1d'], lib="torch", suffix=0)
