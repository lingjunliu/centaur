
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def rand_inputs():
    list_of_inputs = []

    input1 = (2,)
    input_dict1 = {
        "size": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))
    
    input2 = (3, 4)
    input_dict2 = {
        "size": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = (2, 3, 5)
    input_dict3 = {
        "size": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = (4, 2, 3, 2)
    input_dict4 = {
        "size": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = (1, 2, 3, 4, 5)
    input_dict5 = {
        "size": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = rand_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('rand', list_of_inputs)
