
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def MaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    kernel_size1 = 3
    stride1 = 2
    padding1 = (0, 0)
    dilation1 = (1, 1)
    return_indices1 = False
    ceil_mode1 = False
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "return_indices": return_indices1,
        "ceil_mode": ceil_mode1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(5, 3, 25, 25).numpy()
    kernel_size2 = (5, 5)
    stride2 = (2, 2)
    padding2 = (1, 1)
    dilation2 = (1, 1)
    return_indices2 = True
    ceil_mode2 = True
    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "return_indices": return_indices2,
        "ceil_mode": ceil_mode2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 100, 100).numpy()
    kernel_size3 = 7
    stride3 = 7
    padding3 = (3, 3)
    dilation3 = (1, 1)
    return_indices3 = False
    ceil_mode3 = True
    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "return_indices": return_indices3,
        "ceil_mode": ceil_mode3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4, 3, 32, 32).numpy()
    kernel_size4 = (2, 4)
    stride4 = (1, 2)
    padding4 = (0, 1)
    dilation4 = (1, 1)
    return_indices4 = True
    ceil_mode4 = False
    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "return_indices": return_indices4,
        "ceil_mode": ceil_mode4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 64, 64, 64).numpy()
    kernel_size5 = 2
    stride5 = 2
    padding5 = (0, 0)
    dilation5 = (1, 1)
    return_indices5 = False
    ceil_mode5 = False
    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "dilation": dilation5,
        "return_indices": return_indices5,
        "ceil_mode": ceil_mode5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxPool2d_4"] = MaxPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_4'], lib="torch")
