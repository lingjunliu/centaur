
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def ones_inputs():
    list_of_inputs = []

    input1 = {
        "size": [5]
    }
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {
        "size": [2, 3]
    }
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {
        "size": [4, 5, 6]
    }
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {
        "size": [7, 8]
    }
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {
        "size": [1, 2, 3, 4]
    }
    list_of_inputs.append(copy.deepcopy(input5))

    input6 = {
        "size": [1, 1]
    }
    list_of_inputs.append(copy.deepcopy(input6))
    
    return list_of_inputs

generated_inputs = ones_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ones', generated_inputs)
