
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy

def set_grad_enabled_inputs():
    list_of_inputs = []

    input1 = {'set_grad_enabled': True}
    list_of_inputs.append(input1)

    input2 = {'set_grad_enabled': False}
    list_of_inputs.append(input2)

    return list_of_inputs

generated_inputs = set_grad_enabled_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_grad_enabled', generated_inputs)
