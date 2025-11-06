
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def LazyBatchNorm1d_inputs():
    list_of_inputs = []

    input_arr = np.random.randn(4, 3).astype(np.float32)
    input_dict = {
        "eps": 1e-5,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(2, 5, 10).astype(np.float32)
    input_dict = {
        "eps": 1e-3,
        "momentum": 0.9,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(2, 8).astype(np.float32)
    input_dict = {
        "eps": 1e-4,
        "momentum": 0.0,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(16, 1, 7).astype(np.float32)
    input_dict = {
        "eps": 1e-5,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": True,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(32, 128).astype(np.float32)
    input_dict = {
        "eps": 1e-6,
        "momentum": 0.2,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (-5 + 10 * np.random.rand(3, 4)).astype(np.float32)
    input_dict = {
        "eps": 1e-7,
        "momentum": 0.5,
        "affine": False,
        "track_running_stats": False,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.ones((5, 2), dtype=np.float32)
    input_dict = {
        "eps": 1e-5,
        "momentum": 0.8,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (np.random.randn(10, 3) * 100).astype(np.float32)
    input_dict = {
        "eps": 1e-2,
        "momentum": 0.99,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(64, 7, 13).astype(np.float32)
    input_dict = {
        "eps": 1e-5,
        "momentum": 0.01,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    input_dict = {
        "eps": 1e-8,
        "momentum": 0.3,
        "affine": True,
        "track_running_stats": True,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (np.random.randn(6, 9) * 1e-2).astype(np.float32)
    input_dict = {
        "eps": 1e-12,
        "momentum": 0.7,
        "affine": False,
        "track_running_stats": True,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.random.randn(2, 1, 5).astype(np.float32)
    input_dict = {
        "eps": 1e-5,
        "momentum": 0.4,
        "affine": True,
        "track_running_stats": False,
        "dtype": None,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.LazyBatchNorm1d"] = LazyBatchNorm1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LazyBatchNorm1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LazyBatchNorm1d'.")


check_valid('torch.nn.LazyBatchNorm1d', generated_inputs['torch.nn.LazyBatchNorm1d'], lib="torch", suffix=0)
