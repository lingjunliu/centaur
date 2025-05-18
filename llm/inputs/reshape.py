
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def reshape_inputs():
    list_of_inputs = []

    input = torch.randn(2, 3, 4).numpy()
    shape = (24,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(5, 2).numpy()
    shape = (10,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 4, 5).numpy()
    shape = (60,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(10).numpy()
    shape = (10,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2, 2, 2).numpy()
    shape = (16,)
    input_dict = {
        "input": input,
        "shape": shape
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = reshape_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('reshape', list_of_inputs)
