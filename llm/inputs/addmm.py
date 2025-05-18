
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def addmm_inputs():
    list_of_inputs = []

    input = torch.randn(3, 5).numpy()
    mat1 = torch.randn(3, 4).numpy()
    mat2 = torch.randn(4, 5).numpy()
    beta = 1.0
    alpha = 1.0

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3).numpy()
    mat1 = torch.randn(2, 4).numpy()
    mat2 = torch.randn(4, 3).numpy()
    beta = 0.5
    alpha = 2.0

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1).numpy()
    mat1 = torch.randn(1, 5).numpy()
    mat2 = torch.randn(5, 1).numpy()
    beta = 0.0
    alpha = 1.0

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 4).numpy()
    mat1 = torch.randn(4, 2).numpy()
    mat2 = torch.randn(2, 4).numpy()
    beta = -1.0
    alpha = 0.5

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 2).numpy()
    mat1 = torch.randn(5, 3).numpy()
    mat2 = torch.randn(3, 2).numpy()
    beta = 2.0
    alpha = -1.0

    input_dict = {
        "input": input,
        "mat1": mat1,
        "mat2": mat2,
        "beta": beta,
        "alpha": alpha
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = addmm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('addmm', list_of_inputs)
