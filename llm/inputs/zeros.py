
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def zeros_inputs():
    list_of_inputs = []

    input_dict = {
        "size": (5,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (2, 3, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (2, 3, 4, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (2, 3, 4, 5, 6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = zeros_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('zeros', list_of_inputs)
