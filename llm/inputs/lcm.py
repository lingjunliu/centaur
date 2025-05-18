
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def lcm_inputs():
    list_of_inputs = []

    input1 = torch.randint(1, 10, (3, 4)).numpy()
    input2 = torch.randint(1, 10, (3, 4)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randint(1, 10, (2, 2, 2)).numpy()
    input2 = torch.randint(1, 10, (2, 2, 2)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randint(1, 10, (5,)).numpy()
    input2 = torch.randint(1, 10, (5,)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input1 = torch.randint(1, 10, (1, 5, 5)).numpy()
    input2 = torch.randint(1, 10, (1, 5, 5)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input1 = torch.randint(1, 10, (2, 3, 4, 5)).numpy()
    input2 = torch.randint(1, 10, (2, 3, 4, 5)).numpy()
    input_dict = {
        "input": input1,
        "other": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = lcm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lcm', list_of_inputs)
