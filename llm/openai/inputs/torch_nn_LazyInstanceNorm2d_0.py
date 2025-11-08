
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def lazyinstancenorm2d_inputs():
    list_of_inputs = []

    x = (np.random.randn(2, 3, 8, 8)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(3, 8, 8)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-3,
        "momentum": 0.0,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(5, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-6,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(3, 8, 16, 9)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-4,
        "momentum": 0.5,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(4, 1, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-2,
        "momentum": 1.0,
        "affine": False,
        "track_running_stats": True,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(2, 16, 7, 5)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-5,
        "momentum": 0.3,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(6, 5, 4, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-7,
        "momentum": 0.75,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(1, 3, 224, 224)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(8, 7, 6, 6)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 5e-5,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(2, 3, 2, 10)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-1,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": True,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(3, 12, 5, 9)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 1e-8,
        "momentum": 0.6,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": x
    }))

    x = (np.random.randn(1, 4, 1, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "eps": 2e-5,
        "momentum": 0.4,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": x
    }))

    return list_of_inputs

generated_inputs["torch.nn.LazyInstanceNorm2d"] = lazyinstancenorm2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LazyInstanceNorm2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LazyInstanceNorm2d'.")


check_valid('torch.nn.LazyInstanceNorm2d', generated_inputs['torch.nn.LazyInstanceNorm2d'], lib="torch", suffix=0)
