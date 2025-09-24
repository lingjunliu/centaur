
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def fft2_inputs():
    list_of_inputs = []

    # Input 1: Basic complex tensor
    input1 = torch.randn(10, 10, dtype=torch.complex64).numpy()
    input_dict1 = {
        "input": input1,
        "s": None,
        "dim": (-2, -1),
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Real tensor with specified s and dim
    input2 = torch.randn(12, 12).numpy()
    input_dict2 = {
        "input": input2,
        "s": (8, 8),
        "dim": (0, 1),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Real tensor with ortho norm
    input3 = torch.randn(16, 16).numpy()
    input_dict3 = {
        "input": input3,
        "s": None,
        "dim": (-2, -1),
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Different dimensions
    input4 = torch.randn(5, 7, 9, 11).numpy()
    input_dict4 = {
        "input": input4,
        "s": (7,9),
        "dim": (1,2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Integer tensor
    input5 = torch.randint(0, 10, (8, 8)).numpy()
    input_dict5 = {
        "input": input5,
        "s": None,
        "dim": (-2, -1),
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = fft2_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fft2', generated_inputs)
