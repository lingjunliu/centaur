
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def sym_int_inputs():
    list_of_inputs = []

    # Example 1: Positive integer
    input_dict = {"a": np.array(5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Zero
    input_dict = {"a": np.array(0)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Negative integer
    input_dict = {"a": np.array(-3)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Large integer
    input_dict = {"a": np.array(1000)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Small integer
    input_dict = {"a": np.array(-1000)}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = sym_int_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sym_int', generated_inputs)
