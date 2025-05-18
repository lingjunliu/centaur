
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def rot90_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 4, 5).numpy()
    k1 = 1
    dims1 = (2, 3)
    input_dict1 = {
        "input": input1,
        "k": k1,
        "dims": dims1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3, 4).numpy()
    k2 = 2
    dims2 = (1, 2)
    input_dict2 = {
        "input": input2,
        "k": k2,
        "dims": dims2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5, 5).numpy()
    k3 = 3
    dims3 = (0, 1)
    input_dict3 = {
        "input": input3,
        "k": k3,
        "dims": dims3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(1, 1, 10, 10).numpy()
    k4 = -1
    dims4 = (2, 3)
    input_dict4 = {
        "input": input4,
        "k": k4,
        "dims": dims4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3, 5, 7, 9).numpy()
    k5 = 0
    dims5 = (1, 3)
    input_dict5 = {
        "input": input5,
        "k": k5,
        "dims": dims5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

list_of_inputs = rot90_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('rot90', list_of_inputs)
