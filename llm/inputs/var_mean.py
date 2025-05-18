
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def var_mean_inputs():
    list_of_inputs = []

    input = torch.randn(3, 4, 5).numpy()
    dim = 1
    correction = 1
    keepdim = False
    input_dict = {
        "input": input,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2).numpy()
    dim = 0
    correction = 0
    keepdim = True
    input_dict = {
        "input": input,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5).numpy()
    dim = 0
    correction = 1
    keepdim = False
    input_dict = {
        "input": input,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 4).numpy()
    dim = 2
    correction = 2
    keepdim = True
    input_dict = {
        "input": input,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 5, 5).numpy()
    dim = 1
    correction = 1
    keepdim = False
    input_dict = {
        "input": input,
        "dim": dim,
        "correction": correction,
        "keepdim": keepdim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = var_mean_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('var_mean', list_of_inputs)
