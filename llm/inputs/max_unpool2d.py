
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def max_unpool2d_inputs():
    list_of_inputs = []

    # Test case 1: Basic case
    input1 = torch.randn(1, 1, 2, 2).numpy()
    indices1 = torch.tensor([[
        [0, 1],
        [2, 3]
    ]]).numpy().astype(np.int64)
    input_dict1 = {
        "input": input1,
        "indices": indices1,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Different batch size and channel
    input2 = torch.randn(1, 1, 3, 3).numpy()
    indices2 = torch.randint(0, 9, (1, 1, 3, 3)).numpy().astype(np.int64)
    input_dict2 = {
        "input": input2,
        "indices": indices2,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "output_size": (5, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Different kernel_size, stride, padding
    input3 = torch.randn(1, 1, 4, 4).numpy()
    indices3 = torch.randint(0, 16, (1, 1, 4, 4)).numpy().astype(np.int64)
    input_dict3 = {
        "input": input3,
        "indices": indices3,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Output size specified
    input4 = torch.randn(1, 1, 2, 2).numpy()
    indices4 = torch.randint(0, 4, (1, 1, 2, 2)).numpy().astype(np.int64)
    input_dict4 = {
        "input": input4,
        "indices": indices4,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "output_size": (3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Larger input
    input5 = torch.randn(1, 1, 5, 5).numpy()
    indices5 = torch.randint(0, 25, (1, 1, 5, 5)).numpy().astype(np.int64)
    input_dict5 = {
        "input": input5,
        "indices": indices5,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Test case 6: Non-square kernel and stride
    input6 = torch.randn(1, 1, 3, 4).numpy()
    indices6 = torch.randint(0, 12, (1, 1, 3, 4)).numpy().astype(np.int64)
    input_dict6 = {
        "input": input6,
        "indices": indices6,
        "kernel_size": (2, 3),
        "stride": (2, 2),
        "padding": (0, 0),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = max_unpool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('max_unpool2d', generated_inputs)
