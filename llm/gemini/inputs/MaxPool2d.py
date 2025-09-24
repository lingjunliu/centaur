
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def MaxPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 32, 32).numpy()
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

    input2 = torch.randint(0, 10, (2, 1, 16, 16)).numpy()
    kernel_size2 = 2
    stride2 = 1
    padding2 = 0
    dilation2 = 1
    ceil_mode2 = True
    return_indices2 = True

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

    input3 = torch.randn(4, 3, 64, 64).numpy()
    kernel_size3 = (3, 2)
    stride3 = (2, 1)
    padding3 = (1, 0)
    dilation3 = (1, 2)
    ceil_mode3 = False
    return_indices3 = False

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

    input4 = torch.randn(1, 1, 20, 20).numpy()
    kernel_size4 = 5
    stride4 = None
    padding4 = 2
    dilation4 = 1
    ceil_mode4 = False
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

    input5 = torch.randn(2, 3, 10, 10).numpy()
    kernel_size5 = (2, 2)
    stride5 = (2, 2)
    padding5 = (1, 1)
    dilation5 = (1, 1)
    ceil_mode5 = True
    return_indices5 = False

    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "dilation": dilation5,
        "ceil_mode": ceil_mode5,
        "return_indices": return_indices5
    }

    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(1, 1, 7, 7).numpy()
    kernel_size6 = 3
    stride6 = 1
    padding6 = 0
    dilation6 = 2
    ceil_mode6 = False
    return_indices6 = False

    input_dict6 = {
        "input": input6,
        "kernel_size": kernel_size6,
        "stride": stride6,
        "padding": padding6,
        "dilation": dilation6,
        "ceil_mode": ceil_mode6,
        "return_indices": return_indices6
    }

    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(1, 3, 8, 8).numpy()
    kernel_size7 = (2, 3)
    stride7 = None
    padding7 = (1, 0)
    dilation7 = (2, 1)
    ceil_mode7 = True
    return_indices7 = True

    input_dict7 = {
        "input": input7,
        "kernel_size": kernel_size7,
        "stride": stride7,
        "padding": padding7,
        "dilation": dilation7,
        "ceil_mode": ceil_mode7,
        "return_indices": return_indices7
    }

    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = MaxPool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MaxPool2d', generated_inputs)
