
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def MaxPool2d_inputs():
    list_of_inputs = []

    input = torch.randn(20, 16, 50, 32).numpy()
    kernel_size = (3, 2)
    stride = (2, 1)
    padding = (0, 0)
    dilation = (1, 1)
    return_indices = False
    ceil_mode = False
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(10, 8, 25, 16).numpy()
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True
    ceil_mode = True
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 4, 12, 8).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (2, 2)
    return_indices = False
    ceil_mode = False
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2, 60, 40).numpy()
    kernel_size = 5
    stride = None
    padding = 2
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 1, 100, 100).numpy()
    kernel_size = (7, 5)
    stride = (3, 2)
    padding = (1, 1)
    dilation = (3, 2)
    return_indices = True
    ceil_mode = True
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxPool2d_12"] = MaxPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_12'], lib="torch")
