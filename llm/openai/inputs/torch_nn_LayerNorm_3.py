
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def layernorm_inputs_3():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(8, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": (8,),
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True,
        "dtype": torch.float32,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.linspace(-3, 3, steps=24, dtype=torch.float32).reshape(4, 6).numpy()
    input_dict = {
        "normalized_shape": (6,),
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": False,
        "dtype": torch.float32,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = (torch.randn(2, 5, 10, dtype=torch.float32) * 5.0).numpy()
    input_dict = {
        "normalized_shape": (10,),
        "eps": 1e-6,
        "elementwise_affine": False,
        "bias": True,
        "dtype": torch.float32,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(3, 5, 10, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": (5, 10),
        "eps": 1e-4,
        "elementwise_affine": True,
        "bias": True,
        "dtype": torch.float32,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.randn(2, 3, 4, 5, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": (3, 4, 5),
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": False,
        "dtype": torch.float32,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = (torch.randn(2, 3, 4, 5, dtype=torch.float64) * 1000.0).numpy()
    input_dict = {
        "normalized_shape": (5,),
        "eps": 1e-3,
        "elementwise_affine": False,
        "bias": False,
        "dtype": torch.float64,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.randn(2, 2, 3, 4, 5, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": (4, 5),
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True,
        "dtype": torch.float32,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.tensor([0.0], dtype=torch.float64).numpy()
    input_dict = {
        "normalized_shape": (1,),
        "eps": 1e-8,
        "elementwise_affine": False,
        "bias": True,
        "dtype": torch.float64,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.tensor([[42.0]], dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": (1,),
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True,
        "dtype": torch.float32,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16).sub_(8.0).numpy()
    input_dict = {
        "normalized_shape": (16,),
        "eps": 1e-7,
        "elementwise_affine": True,
        "bias": False,
        "dtype": torch.float32,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.randn(6, 2, 3, dtype=torch.float64).numpy()
    input_dict = {
        "normalized_shape": (2, 3),
        "eps": 1e-5,
        "elementwise_affine": False,
        "bias": False,
        "dtype": torch.float64,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.randn(8, 7, 9, dtype=torch.float32).numpy()
    input_dict = {
        "normalized_shape": (9,),
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": False,
        "dtype": torch.float32,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.LayerNorm_3"] = layernorm_inputs_3()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.LayerNorm_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LayerNorm_3'.")


check_valid('torch.nn.LayerNorm', generated_inputs['torch.nn.LayerNorm_3'], lib="torch", suffix=3)
