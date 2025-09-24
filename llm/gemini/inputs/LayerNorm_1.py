
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def layernorm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integer normalized_shape
    input_1 = torch.randn(20, 5, 10).numpy()
    input_dict_1 = {
        "normalized_shape": 10,
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: normalized_shape as a list
    input_2 = torch.randn(20, 5, 10, 10).numpy()
    input_dict_2 = {
        "normalized_shape": [10, 10],
        "eps": 1e-05,
        "elementwise_affine": False,
        "bias": False,
        "dtype": None,
        "input": input_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Different eps value
    input_3 = torch.randn(20, 5, 10).numpy()
    input_dict_3 = {
        "normalized_shape": 10,
        "eps": 1e-08,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Float32 dtype
    input_4 = torch.randn(20, 5, 10).numpy()
    input_dict_4 = {
        "normalized_shape": 10,
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": torch.float32,
        "input": input_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Different input shape and normalized shape
    input_5 = torch.randn(20, 5, 7, 13).numpy()
    input_dict_5 = {
        "normalized_shape": [7, 13],
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LayerNorm_1"] = layernorm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.LayerNorm', generated_inputs['torch.nn.LayerNorm_1'], lib="torch")
