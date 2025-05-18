
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def logcumsumexp_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4, 5).numpy()
    dim1 = 1
    input_dict1 = {
        "input": input1,
        "dim": dim1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 2).numpy()
    dim2 = 0
    input_dict2 = {
        "input": input2,
        "dim": dim2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 5, 7).numpy()
    dim3 = 2
    input_dict3 = {
        "input": input3,
        "dim": dim3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4).numpy()
    dim4 = 0
    input_dict4 = {
        "input": input4,
        "dim": dim4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(2, 3, 4, 5).numpy()
    dim5 = 3
    input_dict5 = {
        "input": input5,
        "dim": dim5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = logcumsumexp_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logcumsumexp', list_of_inputs)
