
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def std_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor, dim=None, correction=1, keepdim=False
    input1 = torch.randn(5).numpy()
    input_dict1 = {
        "input": input1,
        "dim": None,
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor, dim=(0,), correction=0, keepdim=True
    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {
        "input": input2,
        "dim": (0,),
        "correction": 0,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor, dim=(1, 2), correction=2, keepdim=False
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {
        "input": input3,
        "dim": (1, 2),
        "correction": 2,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D float tensor with negative values, dim=(0, 2), correction=1, keepdim=True
    input4 = (torch.randn(2, 3, 2, 3) * -1).numpy()
    input_dict4 = {
        "input": input4,
        "dim": (0, 2),
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D complex tensor, dim=(1,), correction=1, keepdim=False
    input5 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict5 = {
        "input": input5,
        "dim": (1,),
        "correction": 1,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 1D tensor, dim=None, correction=0, keepdim=True
    input6 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).numpy()
    input_dict6 = {
        "input": input6,
        "dim": None,
        "correction": 0,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = std_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('std', generated_inputs)
