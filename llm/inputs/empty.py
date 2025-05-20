
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def torch_empty_inputs():
    list_of_inputs = []

    # Example 1: Basic 2D tensor
    input_dict = {
        "size": (2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 3D tensor
    input_dict = {
        "size": (4, 5, 6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 1D tensor
    input_dict = {
        "size": (10,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Large tensor
    input_dict = {
        "size": (100, 100)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: Different dtypes
    input_dict = {
        "size": (2, 3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: Different layouts
    input_dict = {
        "size": (2, 3),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = torch_empty_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('empty', generated_inputs)
