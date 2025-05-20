
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def native_dropout_inputs():
    list_of_inputs = []

    # Test case 1: Simple 2D float tensor
    input1 = torch.randn(5, 5).numpy()
    input_dict1 = {
        "input": input1,
        "p": 0.5,
        "train": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 3D integer tensor with dropout probability 0 (converted to float)
    input2 = torch.randint(0, 10, (3, 4, 5)).float().numpy()
    input_dict2 = {
        "input": input2,
        "p": 0.0,
        "train": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 1D float tensor with dropout probability 1
    input3 = torch.randn(10).numpy()
    input_dict3 = {
        "input": input3,
        "p": 1.0,
        "train": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Test case 4: 4D float tensor with dropout probability 0.2, train = False
    input4 = torch.randn(2, 3, 4, 5).numpy()
    input_dict4 = {
        "input": input4,
        "p": 0.2,
        "train": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: 2D complex tensor (converted to float)
    input5 = (torch.randn(5, 5) + 1j * torch.randn(5, 5)).abs().numpy()
    input_dict5 = {
        "input": input5,
        "p": 0.3,
        "train": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = native_dropout_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('native_dropout', generated_inputs)
