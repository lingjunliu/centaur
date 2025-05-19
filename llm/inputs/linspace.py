
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def linspace_inputs():
    generated_inputs = []

    start = np.array(0.0, dtype=np.float32)
    end = np.array(10.0, dtype=np.float32)
    steps = 5
    input_dict = {
        "start": torch.tensor(start),
        "end": torch.tensor(end),
        "steps": steps,
        "dtype": torch.float32,
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    start = np.array(-5, dtype=np.int32)
    end = np.array(5, dtype=np.int32)
    steps = 11
    input_dict = {
        "start": torch.tensor(start),
        "end": torch.tensor(end),
        "steps": steps,
        "dtype": torch.int32,
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    start = np.array(1+1j, dtype=np.complex64)
    end = np.array(5+5j, dtype=np.complex64)
    steps = 5
    input_dict = {
        "start": torch.tensor(start),
        "end": torch.tensor(end),
        "steps": steps,
        "dtype": torch.complex64,
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    start = 0.0
    end = 10.0
    steps = 5
    input_dict = {
        "start": torch.tensor(start),
        "end": torch.tensor(end),
        "steps": steps,
        "dtype": torch.float64,
        "requires_grad": True
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    
    start = -5
    end = 5
    steps = 11
    input_dict = {
        "start": torch.tensor(start),
        "end": torch.tensor(end),
        "steps": steps,
        "dtype": torch.int64,
        "requires_grad": False
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = linspace_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('linspace', generated_inputs)
