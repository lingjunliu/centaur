
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import numpy as np
import copy

def max_unpool3d_inputs():
    list_of_inputs = []

    # Case 1: Basic case with N, C, Din, Hin, Win
    pool = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
    input_tensor = torch.randn(2, 3, 10, 10, 10)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different kernel_size and stride
    pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), return_indices=True)
    input_tensor = torch.randn(1, 1, 7, 7, 7)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Non-square kernel and stride
    pool = nn.MaxPool3d(kernel_size=(2, 3, 4), stride=(1, 2, 3), return_indices=True)
    input_tensor = torch.randn(1, 1, 10, 10, 10)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": (2, 3, 4),
        "stride": (1, 2, 3),
        "padding": (0, 0, 0),
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different input size
    pool = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
    input_tensor = torch.randn(1, 1, 5, 5, 5)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Single element batch
    pool = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
    input_tensor = torch.randn(1, 1, 10, 10, 10)
    output, indices = pool(input_tensor)
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = max_unpool3d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MaxUnpool3d', generated_inputs)
