
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def set_autocast_cpu_enabled_inputs():
    list_of_inputs = []

    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "enabled": bool(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "enabled": bool(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = set_autocast_cpu_enabled_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_autocast_cpu_enabled', generated_inputs)
