
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def linspace_inputs():
    list_of_inputs = []

    start = torch.tensor(0.0).item()
    end = torch.tensor(1.0).item()
    steps = 5
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    start = torch.tensor(-1.0).item()
    end = torch.tensor(1.0).item()
    steps = 10
    dtype = torch.float64
    requires_grad = True
    
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    start = torch.tensor(2.0).item()
    end = torch.tensor(5.0).item()
    steps = 7
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    start = torch.tensor(-5.0).item()
    end = torch.tensor(-2.0).item()
    steps = 4
    dtype = torch.float64
    requires_grad = True
    
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    start = torch.tensor(10.0).item()
    end = torch.tensor(20.0).item()
    steps = 12
    dtype = torch.float32
    requires_grad = False
    
    input_dict = {
        "start": start,
        "end": end,
        "steps": steps,
        "dtype": dtype,
        "requires_grad": requires_grad
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

list_of_inputs = linspace_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('linspace', list_of_inputs)
