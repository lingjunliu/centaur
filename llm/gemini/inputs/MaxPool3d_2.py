
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_nn_MaxPool3d_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input1 = torch.randn(2, 3, 10, 10, 10).numpy()
    input_dict1 = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different kernel size, stride, and padding
    input2 = torch.randn(1, 1, 20, 20, 20).numpy()
    input_dict2 = {
        "kernel_size": (5, 5, 5),
        "stride": (3, 3, 3),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Single int for kernel_size, stride, padding, dilation
    input3 = torch.randn(4, 2, 15, 15, 15).numpy()
    input_dict3 = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Asymmetric kernel_size, stride and padding
    input4 = torch.randn(1, 1, 25, 30, 35).numpy()
    input_dict4 = {
        "kernel_size": (3, 4, 5),
        "stride": (1, 2, 3),
        "padding": (1, 1, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Input with small dimensions to test ceil_mode
    input5 = torch.randn(1, 1, 5, 5, 5).numpy()
    input_dict5 = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_2"] = torch_nn_MaxPool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_2'], lib="torch")
