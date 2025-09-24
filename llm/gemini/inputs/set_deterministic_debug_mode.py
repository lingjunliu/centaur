
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def set_deterministic_debug_mode_inputs():
    list_of_inputs = []

    input_dict = {
        "mode": "default"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "mode": "warn"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "mode": "error"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "mode": "off"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "mode": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = set_deterministic_debug_mode_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_deterministic_debug_mode', generated_inputs)
