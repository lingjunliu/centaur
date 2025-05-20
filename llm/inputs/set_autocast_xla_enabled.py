
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def set_autocast_xla_enabled_inputs():
    list_of_inputs = []

    input_dict_1 = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {
        "enabled": bool(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_dict_4 = {
        "enabled": bool(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_dict_5 = {
        "enabled": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_dict_6 = {
        "enabled": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    return list_of_inputs

generated_inputs = set_autocast_xla_enabled_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_autocast_xla_enabled', generated_inputs)
