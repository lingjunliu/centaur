
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def flatten_inputs():
    list_of_inputs = []

    # Example 1: Default parameters, 4D input
    input1 = torch.randn(32, 1, 5, 5).numpy()
    input_dict1 = {
        "start_dim": 1,
        "end_dim": -1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Non-default parameters, 4D input
    input2 = torch.randn(32, 1, 5, 5).numpy()
    input_dict2 = {
        "start_dim": 0,
        "end_dim": 2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: Different input size, 3D input
    input3 = torch.randn(10, 20, 30).numpy()
    input_dict3 = {
        "start_dim": 1,
        "end_dim": 2,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Negative start_dim and end_dim
    input4 = torch.randn(5, 3, 4, 2).numpy()
    input_dict4 = {
        "start_dim": -2,
        "end_dim": -1,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Example 5: Flatten everything, 2D input
    input5 = torch.randn(128, 256).numpy()
    input_dict5 = {
        "start_dim": 0,
        "end_dim": -1,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Example 6: 5D input
    input6 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict6 = {
        "start_dim": 2,
        "end_dim": 4,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Example 7: 1D input
    input7 = torch.randn(100).numpy()
    input_dict7 = {
        "start_dim": 0,
        "end_dim": -1,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Flatten"] = flatten_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.Flatten', generated_inputs['torch.nn.Flatten'], lib="torch")
