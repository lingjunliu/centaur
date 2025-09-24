
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def ldexp_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4).numpy()
    other1 = torch.randint(0, 5, (3, 4)).numpy()
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 2, 2).numpy()
    other2 = torch.randint(-3, 3, (2, 2, 2)).numpy()
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5).numpy()
    other3 = torch.randint(-10, 10, (5,)).numpy()
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 1, 1).numpy()
    other4 = torch.randint(-5, 5, (1, 1, 1, 1)).numpy()
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 3).double().numpy()
    other5 = torch.randint(0, 8, (2, 3)).numpy()
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(4).int().numpy()
    other6 = torch.randint(-2, 2, (4,)).numpy()
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(2, 2).float().numpy()
    other7 = torch.randint(0, 16, (2, 2)).numpy()
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = ldexp_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ldexp_', generated_inputs)
