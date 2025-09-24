
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy

def torch_jit_ignore_inputs():
    list_of_inputs = []

    # Case 1: drop=False, no kwargs
    input_dict = {
        "drop": False,
        "kwargs": ()
    }
    list_of_inputs.append(input_dict)

    # Case 2: drop=True, no kwargs
    input_dict = {
        "drop": True,
        "kwargs": ()
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = torch_jit_ignore_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ignore', generated_inputs)
