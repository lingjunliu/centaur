
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def ones_inputs():
    list_of_inputs = []

    input_dict = {
        "size": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (5,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (10,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "size": (100,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "size": (1000,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = ones_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ones', list_of_inputs)
