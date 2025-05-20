
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def constant_pad1d_inputs():
    list_of_inputs = []

    # Example 1: Integer padding, float value
    input_dict = {
        "input": torch.randn(1, 2, 4).numpy(),
        "padding": 2,
        "value": 3.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Tuple padding, integer value
    input_dict = {
        "input": torch.randn(1, 2, 3).numpy(),
        "padding": (3, 1),
        "value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Zero padding, negative value
    input_dict = {
        "input": torch.randn(1, 1, 5).numpy(),
        "padding": 0,
        "value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Large padding values, large value
    input_dict = {
        "input": torch.randn(1, 3, 2).numpy(),
        "padding": (10, 5),
        "value": 100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Same padding on both sides, negative value
    input_dict = {
        "input": torch.randn(2, 2, 1).numpy(),
        "padding": 5,
        "value": -2.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = constant_pad1d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ConstantPad1d', generated_inputs)
