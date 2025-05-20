
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def sym_int_inputs():
    list_of_inputs = []

    # Input 1: Positive integer
    a1 = 5
    input_dict1 = {"a": a1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative integer
    a2 = -3
    input_dict2 = {"a": a2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Zero
    a3 = 0
    input_dict3 = {"a": a3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Another positive integer
    a4 = 12345
    input_dict4 = {"a": a4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Another negative integer
    a5 = -67890
    input_dict5 = {"a": a5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
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
