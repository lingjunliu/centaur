
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def layernorm_inputs():
    list_of_inputs = []

    # 1
    x = torch.randn(10, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": [10],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    x = torch.randn(4, 8, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": [8],
        "eps": 1e-3,
        "elementwise_affine": True,
        "bias": False,
        "dtype": None,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    x = torch.randn(7, 4, dtype=torch.float64).numpy()
    input_dict = {
        "normalized_shape": [4],
        "eps": 1e-6,
        "elementwise_affine": False,
        "bias": True,
        "dtype": None,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    x = torch.randn(2, 5, 10, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": [10],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    x = (torch.randn(3, 7, 7, dtype=torch.float32) * 3).numpy()
    input_dict = {
        "normalized_shape": [7, 7],
        "eps": 1e-4,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    x = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": [4, 5],
        "eps": 1e-5,
        "elementwise_affine": False,
        "bias": False,
        "dtype": None,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    x = torch.randn(2, 5, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": [10, 10],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    x = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": [4, 5, 6],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": False,
        "dtype": None,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    x = torch.randn(6, dtype=torch.float64).numpy()
    input_dict = {
        "normalized_shape": [6],
        "eps": 1e-9,
        "elementwise_affine": False,
        "bias": False,
        "dtype": None,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    x = torch.randn(8, 2, 16, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": [16],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": x
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.LayerNorm_2"] = layernorm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LayerNorm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LayerNorm_2'.")


check_valid('torch.nn.LayerNorm', generated_inputs['torch.nn.LayerNorm_2'], lib="torch", suffix=2)
