
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def var_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(2, 3, 4).numpy()
    dim = 1
    correction = 0
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(5, 5).numpy()
    dim = 0
    correction = 1
    keepdim = True
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 2, 2).numpy()
    dim = 2
    correction = 0
    keepdim = True
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(10).numpy()
    dim = 0
    correction = 1
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(3, 4, 5, 6).numpy()
    dim = 2
    correction = 0
    keepdim = False
    input_dict = {
        "input": input_tensor,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = var_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('var', list_of_inputs)
