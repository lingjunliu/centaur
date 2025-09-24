
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def fractional_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integer output_size
    input1 = torch.randn(1, 1, 32, 32).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": (3, 3),
        "output_size": (16, 16),
        "output_ratio": None,
        "return_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Basic case with float output_ratio
    input2 = torch.randn(1, 3, 64, 64).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": (2, 2),
        "output_size": None,
        "output_ratio": (0.5, 0.5),
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different kernel size and input size
    input3 = torch.randn(2, 1, 10, 10).numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": (2, 2),
        "output_size": (3,3),
        "output_ratio": None,
        "return_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4:  Non-square input
    input4 = torch.randn(1, 1, 20, 30).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": (2, 2),
        "output_size": (5, 7),
        "output_ratio": None,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Non-square kernel
    input5 = torch.randn(1, 3, 40, 40).numpy()
    input_dict5 = {
        "input": input5,
        "kernel_size": (2, 3),
        "output_size": None,
        "output_ratio": (0.6, 0.6),
        "return_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = fractional_max_pool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('FractionalMaxPool2d', generated_inputs)
