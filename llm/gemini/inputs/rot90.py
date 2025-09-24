
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def rot90_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, k=1, dims=(0, 1)
    input1 = torch.randn(4, 5).numpy()
    input_dict1 = {
        "input": input1,
        "k": 1,
        "dims": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D int tensor, k=2, dims=(1, 2)
    input2 = torch.randint(0, 10, (3, 6, 7)).numpy()
    input_dict2 = {
        "input": input2,
        "k": 2,
        "dims": (1, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D complex tensor, k=-1, dims=(2, 3)
    input3 = torch.randn(2, 3, 4, 5, dtype=torch.complex64).numpy()
    input_dict3 = {
        "input": input3,
        "k": -1,
        "dims": (2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 2D float tensor, k=0, dims=(0, 1)
    input4 = torch.randn(5, 5).numpy()
    input_dict4 = {
        "input": input4,
        "k": 0,
        "dims": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D float tensor, k=3, dims=(0, 2)
    input5 = torch.randn(2, 3, 4).numpy()
    input_dict5 = {
        "input": input5,
        "k": 3,
        "dims": (0, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 5D int tensor, k=-2, dims=(1, 4)
    input6 = torch.randint(0, 10, (1, 2, 3, 4, 5)).numpy()
    input_dict6 = {
        "input": input6,
        "k": -2,
        "dims": (1, 4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 2D bool tensor, k=1, dims=(0, 1)
    input7 = torch.randint(0, 2, (4, 5)).bool().numpy()
    input_dict7 = {
        "input": input7,
        "k": 1,
        "dims": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: 4D tensor
    input8 = torch.randn(2, 3, 4, 5).numpy()
    input_dict8 = {
        "input": input8,
        "k": 1,
        "dims": (0, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
        
    return list_of_inputs

generated_inputs = rot90_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('rot90', generated_inputs)
