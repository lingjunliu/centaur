
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def MaxPool3d_inputs():
    generated_inputs = []

    input1 = torch.randn(1, 3, 32, 32, 32).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    generated_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4, 16, 16, 16).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": True,
        "return_indices": False
    }
    generated_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(0, 10, (1, 1, 64, 64, 64)).numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": 4,
        "stride": 4,
        "padding": 2,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": True
    }
    generated_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 2, 8, 8, 8).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": 2,
        "stride": 2,
        "padding": 1,
        "dilation": 2,
        "ceil_mode": True,
        "return_indices": False
    }
    generated_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(4, 3, 12, 12, 12).numpy()
    input_dict5 = {
        "input": input5,
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": True
    }
    generated_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 1, 5, 5, 5).numpy()
    input_dict6 = {
        "input": input6,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    generated_inputs.append(copy.deepcopy(input_dict6))

    return generated_inputs

generated_inputs = MaxPool3d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MaxPool3d', generated_inputs)
