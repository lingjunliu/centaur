
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def MaxPool3d_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 20, 20, 20).numpy()
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 10, 10, 10).numpy()
    kernel_size = (2, 2, 2)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    dilation = (2, 2, 2)
    return_indices = True
    ceil_mode = True

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2, 15, 15, 15).numpy()
    kernel_size = (3, 2, 3)
    stride = None
    padding = (0, 1, 0)
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input = torch.randn(1, 4, 8, 8, 8).numpy()
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 5, 5, 5).numpy()
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    return_indices = False
    ceil_mode = False

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxPool3d_1"] = MaxPool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_1'], lib="torch")
