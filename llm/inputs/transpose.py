
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def transpose_inputs():
    list_of_inputs = []

    input_tensor = torch.randn(2, 3).numpy()
    dim0 = 0
    dim1 = 1
    input_dict = {
        "input": input_tensor,
        "dim0": dim0,
        "dim1": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(4, 5, 6).numpy()
    dim0 = 1
    dim1 = 2
    input_dict = {
        "input": input_tensor,
        "dim0": dim0,
        "dim1": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(1, 7, 8, 9).numpy()
    dim0 = 2
    dim1 = 3
    input_dict = {
        "input": input_tensor,
        "dim0": dim0,
        "dim1": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_tensor = torch.randn(10,).numpy()
    dim0 = 0
    dim1 = 0
    input_dict = {
        "input": input_tensor,
        "dim0": dim0,
        "dim1": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 2, 2, 2, 2).numpy()
    dim0 = 1
    dim1 = 3
    input_dict = {
        "input": input_tensor,
        "dim0": dim0,
        "dim1": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = transpose_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('transpose', list_of_inputs)
