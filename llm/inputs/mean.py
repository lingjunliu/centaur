
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def mean_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "dim": 1,
        "keepdim": False,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(5, 5).numpy()
    input_dict2 = {
        "input": input2,
        "dim": 0,
        "keepdim": True,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(10).numpy()
    input_dict3 = {
        "input": input3,
        "dim": 0,
        "keepdim": False,
        "dtype": np.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, 2, 2).numpy()
    input_dict4 = {
        "input": input4,
        "dim": 2,
        "keepdim": True,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3, 5).numpy()
    input_dict5 = {
        "input": input5,
        "dim": 1,
        "keepdim": True,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = mean_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('mean', list_of_inputs)
