
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def MaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    kernel_size1 = (3, 2)
    stride1 = (2, 1)
    padding1 = 0
    dilation1 = 1
    return_indices1 = False
    ceil_mode1 = False

    input_dict1 = {
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "return_indices": return_indices1,
        "ceil_mode": ceil_mode1,
        "input": input1
    }

    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 3, 256, 256).numpy()
    kernel_size2 = 2
    stride2 = None
    padding2 = 1
    dilation2 = 1
    return_indices2 = False
    ceil_mode2 = False

    input_dict2 = {
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "return_indices": return_indices2,
        "ceil_mode": ceil_mode2,
        "input": input2
    }

    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(4, 1, 64, 64).numpy()
    kernel_size3 = 3
    stride3 = 2
    padding3 = 0
    dilation3 = 2
    return_indices3 = True
    ceil_mode3 = True

    input_dict3 = {
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "return_indices": return_indices3,
        "ceil_mode": ceil_mode3,
        "input": input3
    }

    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 10, 10).numpy()
    kernel_size4 = (5, 5)
    stride4 = (3, 3)
    padding4 = (2, 2)
    dilation4 = (1, 1)
    return_indices4 = False
    ceil_mode4 = False

    input_dict4 = {
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "return_indices": return_indices4,
        "ceil_mode": ceil_mode4,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3, 3, 32, 32).numpy()
    kernel_size5 = (2, 2)
    stride5 = (1, 1)
    padding5 = (1, 1)
    dilation5 = (2, 2)
    return_indices5 = True
    ceil_mode5 = True

    input_dict5 = {
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "dilation": dilation5,
        "return_indices": return_indices5,
        "ceil_mode": ceil_mode5,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_7"] = MaxPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_7'], lib="torch")
