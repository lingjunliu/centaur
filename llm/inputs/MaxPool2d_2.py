
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_nn_MaxPool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(20, 16, 50, 32).numpy()
    input_dict = {
        'kernel_size': (3, 2),
        'stride': (2, 1),
        'padding': (0, 0),
        'dilation': (1, 1),
        'return_indices': False,
        'ceil_mode': False,
        'input': input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 3, 256, 256).numpy()
    input_dict = {
        'kernel_size': (2, 2),
        'stride': (2, 2),
        'padding': (0, 0),
        'dilation': (1, 1),
        'return_indices': True,
        'ceil_mode': True,
        'input': input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(5, 1, 64, 64).numpy()
    input_dict = {
        'kernel_size': (3, 3),
        'stride': (1, 1),
        'padding': (1, 1),
        'dilation': (1, 1),
        'return_indices': False,
        'ceil_mode': False,
        'input': input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input_tensor = torch.randn(1, 1, 128, 128).numpy()
    input_dict = {
        'kernel_size': (4, 4),
        'stride': (4, 4),
        'padding': (0, 0),
        'dilation': (1, 1),
        'return_indices': True,
        'ceil_mode': False,
        'input': input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(3, 3, 32, 32).numpy()
    input_dict = {
        'kernel_size': (2, 2),
        'stride': (1, 1),
        'padding': (1, 1),
        'dilation': (1, 1),
        'return_indices': False,
        'ceil_mode': True,
        'input': input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxPool2d_2"] = torch_nn_MaxPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_2'], lib="torch")
