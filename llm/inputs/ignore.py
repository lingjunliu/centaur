
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy

def jit_ignore_inputs():
    list_of_inputs = []

    # Case 1: drop=False, no kwargs
    input_dict = {
        "drop": False,
        "kwargs": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: drop=True, no kwargs
    input_dict = {
        "drop": True,
        "kwargs": ()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: drop=False, with kwargs (one int)
    input_dict = {
        "drop": False,
        "kwargs": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: drop=True, with kwargs (one int)
    input_dict = {
        "drop": True,
        "kwargs": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: drop=False, with kwargs (multiple types)
    input_dict = {
        "drop": False,
        "kwargs": (1, "hello", 3.14)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = jit_ignore_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ignore', generated_inputs)
