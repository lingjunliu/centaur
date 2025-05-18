
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def fractional_max_pool2d_inputs():
    list_of_inputs = []

    input = torch.randn(1, 3, 32, 32).numpy()
    kernel_size = (3, 3)
    output_size = (16, 16)
    output_ratio = None
    return_indices = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 64, 64).numpy()
    kernel_size = (2, 2)
    output_size = None
    output_ratio = (0.5, 0.5)
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 4, 128, 128).numpy()
    kernel_size = (4, 4)
    output_size = (32, 32)
    output_ratio = None
    return_indices = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 256, 256).numpy()
    kernel_size = (5, 5)
    output_size = None
    output_ratio = (0.25, 0.25)
    return_indices = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2, 10, 10).numpy()
    kernel_size = (2, 2)
    output_size = (3, 3)
    output_ratio = None
    return_indices = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "output_size": output_size,
        "output_ratio": output_ratio,
        "return_indices": return_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = fractional_max_pool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('FractionalMaxPool2d', list_of_inputs)
