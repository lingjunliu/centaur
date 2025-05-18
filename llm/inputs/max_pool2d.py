
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def max_pool2d_inputs():
    list_of_inputs = []

    input = torch.randn(1, 3, 32, 32).numpy()
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 16, 16).numpy()
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    ceil_mode = False
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 64, 64).numpy()
    kernel_size = 4
    stride = 4
    padding = 0
    dilation = 1
    ceil_mode = True
    return_indices = False
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 5, 28, 28).numpy()
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 2
    ceil_mode = False
    return_indices = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 1, 10, 10).numpy()
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = True
    return_indices = True
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "return_indices": return_indices
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = max_pool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('max_pool2d', list_of_inputs)
