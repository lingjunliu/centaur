
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def local_response_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D input
    input_1 = torch.randn(32, 5, 24, 24).numpy()
    input_dict_1 = {
        "size": 2,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
        "input": input_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 4D input
    input_2 = torch.randn(16, 5, 7, 7, 7, 7).numpy()
    input_dict_2 = {
        "size": 3,
        "alpha": 0.0002,
        "beta": 0.5,
        "k": 2.0,
        "input": input_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Different size and parameters
    input_3 = torch.randn(8, 10, 12, 12).numpy()
    input_dict_3 = {
        "size": 5,
        "alpha": 0.0005,
        "beta": 0.8,
        "k": 0.5,
        "input": input_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Negative values in the input
    input_4 = torch.randn(4, 3, 8, 8).numpy() * -1
    input_dict_4 = {
        "size": 1,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
        "input": input_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Small input size
    input_5 = torch.randn(2, 2, 2, 2).numpy()
    input_dict_5 = {
        "size": 1,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
        "input": input_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Different channel size
    input_6 = torch.randn(1, 16, 4, 4).numpy()
    input_dict_6 = {
        "size": 3,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
        "input": input_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 3D input
    input_7 = torch.randn(5, 7, 9).numpy()
    input_dict_7 = {
        "size": 2,
        "alpha": 0.0001,
        "beta": 0.75,
        "k": 1.0,
        "input": input_7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    return list_of_inputs

generated_inputs["torch.nn.LocalResponseNorm"] = local_response_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.LocalResponseNorm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.LocalResponseNorm'.")

check_valid('torch.nn.LocalResponseNorm', generated_inputs['torch.nn.LocalResponseNorm'], lib="torch")
