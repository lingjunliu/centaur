
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_nn_MaxPool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with square kernel and stride
    input = torch.randn(1, 1, 20, 20).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": 0,
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Non-square kernel and stride
    input = torch.randn(1, 3, 30, 40).numpy()
    input_dict = {
        "kernel_size": (2, 3),
        "stride": (1, 2),
        "padding": 1,
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different padding and dilation
    input = torch.randn(2, 5, 25, 35).numpy()
    input_dict = {
        "kernel_size": (4, 4),
        "stride": (3, 3),
        "padding": 2,
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: ceil_mode=True
    input = torch.randn(1, 1, 15, 15).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": 1,
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: return_indices=True
    input = torch.randn(1, 1, 10, 10).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": 0,
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxPool2d_11"] = torch_nn_MaxPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxPool2d_11' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool2d_11'.")

check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_11'], lib="torch")
