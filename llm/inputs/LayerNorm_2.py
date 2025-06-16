
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_nn_LayerNorm_inputs():
    list_of_inputs = []

    # Case 1: Basic case with int normalized_shape
    input_1 = np.random.randn(20, 5, 10).astype(np.float32)
    input_dict_1 = {
        "normalized_shape": [10],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: normalized_shape as a list
    input_2 = np.random.randn(20, 5, 10, 10).astype(np.float32)
    input_dict_2 = {
        "normalized_shape": [10, 10],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Float input with elementwise_affine=False, bias=False
    input_3 = np.random.randn(20, 5, 10).astype(np.float64)
    input_dict_3 = {
        "normalized_shape": [10],
        "eps": 1e-05,
        "elementwise_affine": False,
        "bias": False,
        "dtype": torch.float64,
        "input": input_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Different eps value
    input_4 = np.random.randn(20, 5, 10).astype(np.float32)
    input_dict_4 = {
        "normalized_shape": [10],
        "eps": 1e-03,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Case 5: Input with different dimensions and normalized_shape
    input_5 = np.random.randn(2, 3, 4, 5, 6).astype(np.float32)
    input_dict_5 = {
        "normalized_shape": [6],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Small input values and different dtype
    input_6 = (np.random.rand(2, 3, 4) * 0.01).astype(np.float16)
    input_dict_6 = {
        "normalized_shape": [4],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": torch.float16,
        "input": input_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: Single dimension input, adjusted normalized_shape
    input_7 = np.random.randn(10).astype(np.float32)
    input_dict_7 = {
        "normalized_shape": [10],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LayerNorm_2"] = torch_nn_LayerNorm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.LayerNorm', generated_inputs['torch.nn.LayerNorm_2'], lib="torch")
