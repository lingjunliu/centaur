
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy

def enable_grad_inputs():
    list_of_inputs = []

    input1 = {
        'orig_func': None,
        'x': torch.randn(1, requires_grad=True).detach().numpy()
    }
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {
        'orig_func': None,
        'x': torch.randn(2, 2, requires_grad=True).detach().numpy()
    }
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {
        'orig_func': None,
        'x': torch.randn(3, 3, 3, requires_grad=True).detach().numpy()
    }
    list_of_inputs.append(copy.deepcopy(input3))

    input4 = {
        'orig_func': None,
        'x': torch.randn(1, 1, 1, 1, requires_grad=True).detach().numpy()
    }
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {
        'orig_func': None,
        'x': torch.randn(10, requires_grad=True).detach().numpy()
    }
    list_of_inputs.append(copy.deepcopy(input5))

    return list_of_inputs

generated_inputs = enable_grad_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('enable_grad', generated_inputs)
