
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_nn_MaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 32, 32).numpy()
    input_dict1 = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 1, 28, 28).numpy()
    input_dict2 = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 3, 64, 64).numpy()
    input_dict3 = {
        "kernel_size": (4, 4),
        "stride": (4, 4),
        "padding": (0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 16, 16).numpy()
    input_dict4 = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(4, 3, 128, 128).numpy()
    input_dict5 = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxPool2d_10"] = torch_nn_MaxPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_10'], lib="torch")
