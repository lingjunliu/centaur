
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def trapz_inputs():
    list_of_inputs = []

    y = torch.randn(5).numpy()
    x = torch.arange(0, 5).numpy()
    dx = 1.0
    dim = 0
    input_dict = {
        "y": y,
        "x": x,
        "dx": dx,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = torch.randn(3, 5).numpy()
    x = torch.arange(0, 5).numpy()
    dx = 1.0
    dim = 1
    input_dict = {
        "y": y,
        "x": x,
        "dx": dx,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = torch.randn(5, 3).numpy()
    x = torch.arange(0, 5).numpy()
    dx = 2.0
    dim = 0
    input_dict = {
        "y": y,
        "x": x,
        "dx": dx,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    y = torch.randn(5).numpy()
    x = torch.arange(0, len(y)).numpy()
    dx = 2.0
    dim = 0
    input_dict = {
        "y": y,
        "x": x,
        "dx": dx,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    y = torch.randn(3, 5, 2).numpy()
    x = torch.arange(0, 5).numpy()
    dx = 0.5
    dim = 1
    input_dict = {
        "y": y,
        "x": x,
        "dx": dx,
        "dim": dim
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = trapz_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('trapz', list_of_inputs)
