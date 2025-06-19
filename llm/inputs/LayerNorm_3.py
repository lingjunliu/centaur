
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def LayerNorm_inputs():
    list_of_inputs = []

    # Input 1: Basic case with a single integer normalized_shape
    input1 = torch.randn(20, 10).numpy()
    input_dict1 = {
        "normalized_shape": (10,),
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2:  normalized_shape as a tuple
    input2 = torch.randn(2, 3, 5, 5).numpy()
    input_dict2 = {
        "normalized_shape": (5, 5),
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3:  elementwise_affine=False and bias=False
    input3 = torch.randn(5, 4, 6).numpy()
    input_dict3 = {
        "normalized_shape": (6,),
        "eps": 1e-05,
        "elementwise_affine": False,
        "bias": False,
        "dtype": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4:  different eps value
    input4 = torch.randn(10, 20, 30).numpy()
    input_dict4 = {
        "normalized_shape": (30,),
        "eps": 1e-03,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5:  normalized_shape as a list
    input5 = torch.randn(4, 5, 7, 7).numpy()
    input_dict5 = {
        "normalized_shape": (7, 7),
        "eps": 1e-05,
        "elementwise_affine": True,
        "bias": True,
        "dtype": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LayerNorm_3"] = LayerNorm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.LayerNorm', generated_inputs['torch.nn.LayerNorm_3'], lib="torch")
