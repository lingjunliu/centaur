
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy

def no_grad_inputs():
    list_of_inputs = []

    def dummy_func():
        pass

    input_dict1 = {
        "orig_func": dummy_func
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = no_grad_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('no_grad', generated_inputs)
