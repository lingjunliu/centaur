
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def adaptive_avg_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic float input with a single integer output size
    input1 = torch.randn(1, 3, 32, 32).numpy()
    output_size1 = 16
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different input size, tuple output size
    input2 = torch.randn(2, 4, 64, 64).numpy()
    output_size2 = (8, 8)
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Single channel input
    input3 = torch.randn(1, 1, 128, 128).numpy()
    output_size3 = (32, 32)
    input_dict3 = {"input": input3, "output_size": output_size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Large input size, single int
    input6 = torch.randn(4, 16, 256, 256).numpy()
    output_size6 = 64
    input_dict6 = {"input": input6, "output_size": output_size6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 5: 3D input
    input7 = torch.randn(3, 32, 32).numpy()
    output_size7 = (16, 16)
    input_dict7 = {"input": input7, "output_size": output_size7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = adaptive_avg_pool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('adaptive_avg_pool2d', generated_inputs)
