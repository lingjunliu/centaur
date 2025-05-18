
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def lstsq_inputs():
    list_of_inputs = []

    A = torch.randn(5, 3).numpy()
    B = torch.randn(5, 2).numpy()
    rcond = 1e-15
    input_dict = {
        "A": A,
        "B": B,
        "rcond": rcond
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(10, 5).numpy()
    B = torch.randn(10, 1).numpy()
    rcond = 1e-10
    input_dict = {
        "A": A,
        "B": B,
        "rcond": rcond
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(7, 4).numpy()
    B = torch.randn(7, 3).numpy()
    rcond = 1e-5
    input_dict = {
        "A": A,
        "B": B,
        "rcond": rcond
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(3, 2).numpy()
    B = torch.randn(3, 1).numpy()
    rcond = 1e-2
    input_dict = {
        "A": A,
        "B": B,
        "rcond": rcond
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = torch.randn(6, 4).numpy()
    B = torch.randn(6, 2).numpy()
    rcond = 0.1
    input_dict = {
        "A": A,
        "B": B,
        "rcond": rcond
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = lstsq_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lstsq', list_of_inputs)
