
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def max_pool1d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 10).numpy()
    kernel_size1 = 3
    stride1 = 2
    padding1 = 1
    dilation1 = 1
    ceil_mode1 = False
    return_indices1 = False
    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "dilation": dilation1,
        "ceil_mode": ceil_mode1,
        "return_indices": return_indices1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 5, 15).numpy()
    kernel_size2 = 4
    stride2 = 3
    padding2 = 0
    dilation2 = 2
    ceil_mode2 = True
    return_indices2 = False
    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "dilation": dilation2,
        "ceil_mode": ceil_mode2,
        "return_indices": return_indices2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 20).numpy()
    kernel_size3 = 5
    stride3 = 1
    padding3 = 2
    dilation3 = 1
    ceil_mode3 = False
    return_indices3 = True
    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "dilation": dilation3,
        "ceil_mode": ceil_mode3,
        "return_indices": return_indices3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(3, 2, 12).numpy()
    kernel_size4 = 2
    stride4 = 2
    padding4 = 0
    dilation4 = 1
    ceil_mode4 = True
    return_indices4 = False
    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "dilation": dilation4,
        "ceil_mode": ceil_mode4,
        "return_indices": return_indices4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 4, 8).numpy()
    kernel_size5 = 2
    stride5 = None
    padding5 = 1
    dilation5 = 1
    ceil_mode5 = False
    return_indices5 = False
    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": kernel_size5,
        "padding": padding5,
        "dilation": dilation5,
        "ceil_mode": ceil_mode5,
        "return_indices": return_indices5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = max_pool1d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('max_pool1d', generated_inputs)
