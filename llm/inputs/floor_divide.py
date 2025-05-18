
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def floor_divide_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randn(3, 4).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(10, (2, 2)).numpy()
    other2 = np.array([2]).astype(input2.dtype)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5).numpy()
    other3 = torch.randn(5).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randint(1, 10, (2,3,4)).numpy()
    other4 = np.array([3]).astype(input4.dtype)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.tensor([5.0, 2.0, 3.0]).numpy()
    other5 = torch.tensor([2.0, 2.0, 2.0]).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

list_of_inputs = floor_divide_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('floor_divide', list_of_inputs)
