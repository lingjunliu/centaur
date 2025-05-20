
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy

def jit_error_inputs():
    list_of_inputs = []

    input_dict_1 = {
        "msg": "This is a test error message."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_dict_2 = {
        "msg": "Another error occurred during compilation."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_dict_3 = {
        "msg": "Type mismatch in compiled code."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_dict_4 = {
        "msg": "Index out of bounds."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    input_dict_5 = {
        "msg": "Division by zero error."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_dict_6 = {
        "msg": "Unexpected keyword argument."
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_dict_7 = {
        "msg": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    return list_of_inputs

generated_inputs = jit_error_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Error', generated_inputs)
