
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def groupnorm_inputs():
    list_of_inputs = []

    input_arr = np.random.randn(4, 6, 8, 8).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 3,
        "num_channels": 6,
        "eps": 1e-5,
        "affine": True,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.random.randn(2, 6, 5, 7).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 6,
        "num_channels": 6,
        "eps": 1e-4,
        "affine": False,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.random.randn(5, 6, 1, 1).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 1,
        "num_channels": 6,
        "eps": 1e-3,
        "affine": False,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.linspace(-3.0, 3.0, num=12, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 2,
        "num_channels": 4,
        "eps": 1e-8,
        "affine": True,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.random.randn(2, 8, 100).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 4,
        "num_channels": 8,
        "eps": 1e-5,
        "affine": False,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.random.randn(1, 12, 3, 5, 5).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 3,
        "num_channels": 12,
        "eps": 1e-6,
        "affine": True,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.random.randn(10, 2).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 2,
        "num_channels": 2,
        "eps": 1e-5,
        "affine": False,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.random.randn(3, 9, 9, 9).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 3,
        "num_channels": 9,
        "eps": 1e-2,
        "affine": True,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.random.randn(7, 14, 17).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 7,
        "num_channels": 14,
        "eps": 1e-5,
        "affine": True,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.random.randn(2, 16, 2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 8,
        "num_channels": 16,
        "eps": 1e-7,
        "affine": False,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.random.randn(1, 1, 10, 10).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 1,
        "num_channels": 1,
        "eps": 0.0,
        "affine": True,
        "dtype": None,
        "input": input_arr
    }))

    input_arr = np.random.uniform(-1.0, 1.0, size=(4, 20)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({
        "num_groups": 5,
        "num_channels": 20,
        "eps": 1e-4,
        "affine": True,
        "dtype": None,
        "input": input_arr
    }))

    return list_of_inputs

generated_inputs["torch.nn.GroupNorm"] = groupnorm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.GroupNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.GroupNorm'.")


check_valid('torch.nn.GroupNorm', generated_inputs['torch.nn.GroupNorm'], lib="torch", suffix=0)
