
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def zeros_inputs():
    generated_inputs = []

    input1 = {
        "size": (5,)
    }
    generated_inputs.append(copy.deepcopy(input1))

    input2 = {
        "size": (2, 3)
    }
    generated_inputs.append(copy.deepcopy(input2))

    input3 = {
        "size": (2, 3, 4)
    }
    generated_inputs.append(copy.deepcopy(input3))
    
    input4 = {
        "size": (1, 1, 1, 1)
    }
    generated_inputs.append(copy.deepcopy(input4))

    input5 = {
        "size": (6, 7)
    }
    generated_inputs.append(copy.deepcopy(input5))

    return generated_inputs

generated_inputs = zeros_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('zeros', generated_inputs)
