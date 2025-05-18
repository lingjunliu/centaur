
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def logspace_inputs():
    list_of_inputs = []

    start = torch.tensor(0.0).numpy()
    end = torch.tensor(5.0).numpy()
    steps = 10
    base = 10.0
    dtype = torch.float32
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

    start = torch.tensor(-2.0).numpy()
    end = torch.tensor(2.0).numpy()
    steps = 5
    base = 2.0
    dtype = torch.float64
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

    start = torch.tensor(1.0).numpy()
    end = torch.tensor(10.0).numpy()
    steps = 20
    base = np.e  # Euler's number
    dtype = torch.float32
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

    start = torch.tensor(-1.0).numpy()
    end = torch.tensor(1.0).numpy()
    steps = 15
    base = 5.0
    dtype = torch.float64
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

    start = torch.tensor(2.0).numpy()
    end = torch.tensor(8.0).numpy()
    steps = 7
    base = 3.0
    dtype = torch.float32
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
    
    return list_of_inputs

list_of_inputs = logspace_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logspace', list_of_inputs)
