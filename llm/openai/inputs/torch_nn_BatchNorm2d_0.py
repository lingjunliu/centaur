
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def batchnorm2d_inputs():
    list_of_inputs = []

    # Input 1
    dtype = None
    arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    dtype = None
    arr = np.random.randn(8, 64, 7, 7).astype(np.float32)
    input_dict = {
        "num_features": 64,
        "eps": 1e-3,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": True,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    dtype = None
    arr = np.random.randn(1, 1, 1, 2).astype(np.float32)
    input_dict = {
        "num_features": 1,
        "eps": 1e-6,
        "momentum": 0.0,
        "affine": False,
        "track_running_stats": False,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    dtype = None
    arr = np.random.randn(4, 10, 32, 32).astype(np.float32)
    input_dict = {
        "num_features": 10,
        "eps": 1e-4,
        "momentum": 0.5,
        "affine": True,
        "track_running_stats": False,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    dtype = None
    arr = np.random.randn(16, 32, 8, 12).astype(np.float32)
    input_dict = {
        "num_features": 32,
        "eps": 1e-5,
        "momentum": 0.01,
        "affine": False,
        "track_running_stats": True,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    dtype = None
    arr = np.random.randn(3, 5, 13, 9).astype(np.float32)
    input_dict = {
        "num_features": 5,
        "eps": 1e-2,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": True,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    dtype = None
    arr = np.random.randn(2, 128, 2, 2).astype(np.float32)
    input_dict = {
        "num_features": 128,
        "eps": 1e-8,
        "momentum": 0.7,
        "affine": True,
        "track_running_stats": False,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    dtype = None
    arr = np.random.randn(6, 7, 15, 20).astype(np.float32)
    input_dict = {
        "num_features": 7,
        "eps": 5e-5,
        "momentum": 0.3,
        "affine": False,
        "track_running_stats": True,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    dtype = None
    arr = np.random.randn(5, 3, 224, 224).astype(np.float32)
    input_dict = {
        "num_features": 3,
        "eps": 1e-5,
        "momentum": 0.8,
        "affine": True,
        "track_running_stats": True,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    dtype = None
    arr = np.random.randn(10, 11, 5, 3).astype(np.float32)
    input_dict = {
        "num_features": 11,
        "eps": 1e-1,
        "momentum": 0.4,
        "affine": False,
        "track_running_stats": False,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    dtype = None
    arr = np.random.randn(2, 256, 1, 3).astype(np.float32)
    input_dict = {
        "num_features": 256,
        "eps": 1e-5,
        "momentum": 1.0,
        "affine": True,
        "track_running_stats": True,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    dtype = None
    arr = np.random.randn(7, 9, 17, 1).astype(np.float32)
    input_dict = {
        "num_features": 9,
        "eps": 1e-7,
        "momentum": 0.6,
        "affine": True,
        "track_running_stats": False,
        "dtype": dtype,
        "input": arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.BatchNorm2d"] = batchnorm2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.BatchNorm2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.BatchNorm2d'.")


check_valid('torch.nn.BatchNorm2d', generated_inputs['torch.nn.BatchNorm2d'], lib="torch", suffix=0)
