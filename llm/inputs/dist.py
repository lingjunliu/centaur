
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def dist_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4).numpy()
    input2 = torch.randn(3, 4).numpy()
    p1 = 2.0
    input_dict1 = {
        "input": input1,
        "other": input2,
        "p": p1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input3 = torch.randn(5, 5).numpy()
    input4 = torch.randn(5, 5).numpy()
    p2 = 1.0
    input_dict2 = {
        "input": input3,
        "other": input4,
        "p": p2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input5 = torch.randn(2, 2, 2).numpy()
    input6 = torch.randn(2, 2, 2).numpy()
    p3 = 0.5
    input_dict3 = {
        "input": input5,
        "other": input6,
        "p": p3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input7 = torch.randn(10).numpy()
    input8 = torch.randn(10).numpy()
    p4 = float('inf')
    input_dict4 = {
        "input": input7,
        "other": input8,
        "p": p4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input9 = torch.randn(1, 5).numpy()
    input10 = torch.randn(1, 5).numpy()
    p5 = 3.0
    input_dict5 = {
        "input": input9,
        "other": input10,
        "p": p5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

list_of_inputs = dist_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('dist', list_of_inputs)
