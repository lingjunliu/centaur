
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def solve_inputs():
    list_of_inputs = []

    A = torch.randn(3, 3).numpy()
    B = torch.randn(3, 3).numpy()
    left = True
    input_dict = {
        "A": A,
        "B": B,
        "left": left
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(5, 5).numpy()
    B = torch.randn(5, 5).numpy()
    left = False
    input_dict = {
        "A": A,
        "B": B,
        "left": left
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(2, 2).numpy()
    B = torch.randn(2, 2).numpy()
    left = True
    input_dict = {
        "A": A,
        "B": B,
        "left": left
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    A = torch.randn(4, 4).numpy()
    B = torch.randn(4, 4).numpy()
    left = False
    input_dict = {
        "A": A,
        "B": B,
        "left": left
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(6, 6).numpy()
    B = torch.randn(6, 6).numpy()
    left = True
    input_dict = {
        "A": A,
        "B": B,
        "left": left
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = solve_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('solve', list_of_inputs)
