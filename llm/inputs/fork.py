
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def fork_inputs():
    list_of_inputs = []

    # Define a simple function to be forked
    def simple_func(a, b=1):
        return a + b

    # Input 1: Fork a free function with positional arguments
    input_dict = {
        "func": simple_func,
        "args": [torch.tensor(1)],
        "kwargs": {}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Fork a free function with positional and keyword arguments
    input_dict = {
        "func": simple_func,
        "args": [torch.tensor(2)],
        "kwargs": {"b": torch.tensor(3)}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Fork a free function with no arguments
    def no_arg_func():
        return torch.tensor(4)
    input_dict = {
        "func": no_arg_func,
        "args": [],
        "kwargs": {}
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = fork_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fork', generated_inputs)
