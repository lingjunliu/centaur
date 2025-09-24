
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch
import numpy as np
import copy

def adaptive_max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with 4D tensor
    input1 = torch.randn(1, 3, 32, 32).numpy()
    output_size1 = (16, 16)
    return_indices1 = False
    input_dict1 = {
        "input": input1,
        "output_size": output_size1,
        "return_indices": return_indices1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor
    input2 = torch.randn(3, 32, 32).numpy()
    output_size2 = (16, 16)
    return_indices2 = True
    input_dict2 = {
        "input": input2,
        "output_size": output_size2,
        "return_indices": return_indices2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different output size (single integer)
    input3 = torch.randn(1, 3, 64, 64).numpy()
    output_size3 = (8, 8)
    return_indices3 = False
    input_dict3 = {
        "input": input3,
        "output_size": output_size3,
        "return_indices": return_indices3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Small input size
    input4 = torch.randn(1, 1, 4, 4).numpy()
    output_size4 = (2, 2)
    return_indices4 = True
    input_dict4 = {
        "input": input4,
        "output_size": output_size4,
        "return_indices": return_indices4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Batch size > 1
    input5 = torch.randn(4, 3, 32, 32).numpy()
    output_size5 = (8, 8)
    return_indices5 = False
    input_dict5 = {
        "input": input5,
        "output_size": output_size5,
        "return_indices": return_indices5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Input with different data type (float64)
    input6 = torch.randn(1, 3, 32, 32, dtype=torch.float64).numpy()
    output_size6 = (16, 16)
    return_indices6 = False
    input_dict6 = {
        "input": input6,
        "output_size": output_size6,
        "return_indices": return_indices6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.adaptive_max_pool2d"] = adaptive_max_pool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('adaptive_max_pool2d', generated_inputs['torch.nn.functional.adaptive_max_pool2d'], lib="torch")
