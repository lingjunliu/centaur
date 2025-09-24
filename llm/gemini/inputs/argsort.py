
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def argsort_inputs():
    list_of_inputs = []

    input1 = torch.randn(4, 4).numpy()
    input_dict1 = {
        "input": input1,
        "dim": 1,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(0, 10, (3, 5)).numpy()
    input_dict2 = {
        "input": input2,
        "dim": 0,
        "descending": True,
        "stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {
        "input": input3,
        "dim": 2,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(5).numpy()
    input_dict4 = {
        "input": input4,
        "dim": 0,
        "descending": True,
        "stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2).numpy()
    input_dict5 = {
        "input": input5,
        "dim": 1,
        "descending": False,
        "stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randint(-5, 5, (2, 3)).numpy()
    input_dict6 = {
        "input": input6,
        "dim": 0,
        "descending": True,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(1, 1, 1, 1).numpy()
    input_dict7 = {
        "input": input7,
        "dim": 3,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.randn(2, 2, 2, 2).numpy()
    input_dict8 = {
        "input": input8,
        "dim": 1,
        "descending": True,
        "stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = torch.randn(size=(0,)).numpy()
    input_dict9 = {
        "input": input9,
        "dim": -1,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs = argsort_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('argsort', generated_inputs)
