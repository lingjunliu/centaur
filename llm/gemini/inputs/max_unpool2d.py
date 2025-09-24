
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def max_unpool2d_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with specified output size
    input1 = torch.randn(1, 1, 2, 2).numpy()
    indices1 = torch.randint(0, 4, (1, 1, 2, 2)).numpy()
    input_dict1 = {
        "input": input1,
        "indices": indices1,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "output_size": (4, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Different kernel size and stride
    input2 = torch.randn(1, 1, 3, 3).numpy()
    indices2 = torch.randint(0, 9, (1, 1, 3, 3)).numpy()
    input_dict2 = {
        "input": input2,
        "indices": indices2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Multiple channels
    input3 = torch.randn(1, 2, 2, 2).numpy()
    indices3 = torch.randint(0, 16, (1, 2, 2, 2)).numpy()
    input_dict3 = {
        "input": input3,
        "indices": indices3,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Batch size > 1
    input4 = torch.randn(2, 1, 2, 2).numpy()
    indices4 = torch.randint(0, 16, (2, 1, 2, 2)).numpy()
    input_dict4 = {
        "input": input4,
        "indices": indices4,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Different padding
    input5 = torch.randn(1, 1, 2, 2).numpy()
    indices5 = torch.randint(0, 9, (1, 1, 2, 2)).numpy()
    input_dict5 = {
        "input": input5,
        "indices": indices5,
        "kernel_size": 3,
        "stride": 1,
        "padding": 2,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
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
