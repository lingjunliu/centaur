
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def max_pool3d_inputs():
    list_of_inputs = []

    input1 = torch.randn(1, 3, 10, 10, 10).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(0, 10, (2, 4, 8, 8, 8)).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "ceil_mode": True,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 7, 7, 7).numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": 2,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "ceil_mode": False,
        "return_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 7, 7, 7).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": (2, 3, 2),
        "stride": (1, 2, 1),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "ceil_mode": True,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 2, 9, 9, 9).numpy()
    input_dict5 = {
        "input": input5,
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = max_pool3d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('max_pool3d', generated_inputs)
