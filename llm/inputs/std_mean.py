
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def std_mean_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    dim_val = 1
    correction_val = 0
    keepdim_val = False
    input_dict = {
        "input": input_tensor,
        "dim": dim_val,
        "correction": correction_val,
        "keepdim": keepdim_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(5, 5).numpy()
    dim_val = 0
    correction_val = 1
    keepdim_val = True
    input_dict = {
        "input": input_tensor,
        "dim": dim_val,
        "correction": correction_val,
        "keepdim": keepdim_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(2, 2, 2).numpy()
    dim_val = 2
    correction_val = 2
    keepdim_val = False
    input_dict = {
        "input": input_tensor,
        "dim": dim_val,
        "correction": correction_val,
        "keepdim": keepdim_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(10).numpy()
    dim_val = 0
    correction_val = 0
    keepdim_val = True
    input_dict = {
        "input": input_tensor,
        "dim": dim_val,
        "correction": correction_val,
        "keepdim": keepdim_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 5, 1, 5).numpy()
    dim_val = 3
    correction_val = 1
    keepdim_val = False
    input_dict = {
        "input": input_tensor,
        "dim": dim_val,
        "correction": correction_val,
        "keepdim": keepdim_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = std_mean_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('std_mean', list_of_inputs)
