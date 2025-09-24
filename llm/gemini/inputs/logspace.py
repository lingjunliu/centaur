
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def logspace_inputs():
    list_of_inputs = []

    start = np.array(1.0)
    end = np.array(10.0)
    steps = 5
    base = 10.0
    dtype = torch.float64
    requires_grad = False

    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "base": base,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    start = np.array(0.1)
    end = np.array(100.0)
    steps = 7
    base = 2.0
    dtype = torch.float32
    requires_grad = True

    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "base": base,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    start = np.array(-1.0)
    end = np.array(1.0)
    steps = 6
    base = 5.0
    dtype = torch.float64
    requires_grad = False

    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "base": base,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    start = np.array(0.0)
    end = np.array(5.0)
    steps = 8
    base = 3.0
    dtype = torch.float32
    requires_grad = True

    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "base": base,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    start = np.array(-2.0)
    end = np.array(2.0)
    steps = 9
    base = 10.0
    dtype = torch.float32
    requires_grad = True

    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "base": base,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = logspace_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logspace', generated_inputs)
