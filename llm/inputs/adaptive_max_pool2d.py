
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def adaptive_max_pool2d_inputs():
    list_of_inputs = []

    input = torch.randn(1, 3, 32, 32).numpy()
    output_size = 7
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 4, 64, 64).numpy()
    output_size = 14
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 128, 128).numpy()
    output_size = 28
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2, 256, 256).numpy()
    output_size = 56
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 8, 512, 512).numpy()
    output_size = 112
    input_dict = {
        "input": input,
        "output_size": output_size
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = adaptive_max_pool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('adaptive_max_pool2d', list_of_inputs)
