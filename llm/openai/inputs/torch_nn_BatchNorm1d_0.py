
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def batchnorm1d_inputs():
    list_of_inputs = []

    x = np.random.randn(4, 3).astype(np.float32)
    input_dict = {
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(8, 5, 10).astype(np.float32)
    input_dict = {
        "num_features": 5,
        "eps": 1e-4,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(2, 1).astype(np.float64)
    input_dict = {
        "num_features": 1,
        "eps": 1e-3,
        "momentum": 0.0,
        "affine": False,
        "track_running_stats": True,
        "dtype": torch.float64,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(3, 7, 1).astype(np.float64)
    input_dict = {
        "num_features": 7,
        "eps": 1e-8,
        "momentum": 0.5,
        "affine": True,
        "track_running_stats": False,
        "dtype": torch.float64,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(-10, 10, 8, dtype=np.float64).reshape(2, 4)
    input_dict = {
        "num_features": 4,
        "eps": 1e-5,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "dtype": torch.float64,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.random.randn(16, 2, 20) * 2.0).astype(np.float32)
    input_dict = {
        "num_features": 2,
        "eps": 1e-3,
        "momentum": 0.05,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.uniform(-1.0, 1.0, size=(5, 10)).astype(np.float32)
    input_dict = {
        "num_features": 10,
        "eps": 1e-2,
        "momentum": 0.99,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(6, 3, 7).astype(np.float32)
    input_dict = {
        "num_features": 3,
        "eps": 1e-7,
        "momentum": 0.1,
        "affine": False,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.random.randn(32, 64) * 0.1).astype(np.float32)
    input_dict = {
        "num_features": 64,
        "eps": 5e-4,
        "momentum": 0.01,
        "affine": True,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(2, 2, 4).astype(np.float64)
    input_dict = {
        "num_features": 2,
        "eps": 1e-6,
        "momentum": 1.0,
        "affine": False,
        "track_running_stats": True,
        "dtype": torch.float64,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.random.randn(10, 6) * 100.0).astype(np.float32)
    input_dict = {
        "num_features": 6,
        "eps": 1e-4,
        "momentum": 0.3,
        "affine": True,
        "track_running_stats": False,
        "dtype": torch.float32,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.random.randn(3, 12, 5).astype(np.float32)
    input_dict = {
        "num_features": 12,
        "eps": 1e-5,
        "momentum": 0.7,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float32,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.BatchNorm1d"] = batchnorm1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BatchNorm1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BatchNorm1d'.")


check_valid('torch.nn.BatchNorm1d', generated_inputs['torch.nn.BatchNorm1d'], lib="torch", suffix=0)
