
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def transpose_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor
    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {
        "input": input1,
        "dim0": 0,
        "dim1": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D int tensor
    input2 = torch.randint(0, 10, (2, 3, 4)).numpy()
    input_dict2 = {
        "input": input2,
        "dim0": 1,
        "dim1": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D tensor with negative dimensions
    input3 = torch.randn(2, 3, 4, 5).numpy()
    input_dict3 = {
        "input": input3,
        "dim0": 0,
        "dim1": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor (should still work, though effectively a no-op)
    input4 = torch.randn(5).numpy()
    input_dict4 = {
        "input": input4,
        "dim0": 0,
        "dim1": 0  # No effect since only one dimension
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex tensor
    input5 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict5 = {
        "input": input5,
        "dim0": 0,
        "dim1": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6:  Another 3D tensor with different dimensions to test dim swap
    input6 = torch.randn(5, 2, 3).numpy()
    input_dict6 = {
        "input": input6,
        "dim0": 0,
        "dim1": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7:  Double tensor
    input7 = torch.randn(4, 2, dtype=torch.float64).numpy()
    input_dict7 = {
        "input": input7,
        "dim0": 0,
        "dim1": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = transpose_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('transpose', generated_inputs)
