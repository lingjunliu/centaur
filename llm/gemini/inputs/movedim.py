
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def movedim_inputs():
    list_of_inputs = []

    # Input 1: 2D tensor, move axis 0 to 1
    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {
        "input": input1,
        "source": 0,
        "destination": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor, move axis 1 to 0
    input2 = torch.randint(0, 10, (2, 3, 4)).numpy()
    input_dict2 = {
        "input": input2,
        "source": 1,
        "destination": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D tensor, move axis 3 to -1 (last)
    input3 = torch.randn(2, 3, 4, 5).numpy()
    input_dict3 = {
        "input": input3,
        "source": 3,
        "destination": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 5D tensor, move axis -2 to 0
    input4 = torch.randn(2, 3, 4, 5, 6).numpy()
    input_dict4 = {
        "input": input4,
        "source": -2,
        "destination": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensor, move axis 0 to -1
    input5 = torch.randn(3, 4, 5).numpy()
    input_dict5 = {
        "input": input5,
        "source": 0,
        "destination": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 1D tensor, move axis 0 to 0 (no change, but valid)
    input6 = torch.arange(5).numpy()
    input_dict6 = {
        "input": input6,
        "source": 0,
        "destination": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 3D tensor, move axis -1 to 1
    input7 = torch.randn(2, 3, 4).numpy()
    input_dict7 = {
        "input": input7,
        "source": -1,
        "destination": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Complex tensor, move axis 0 to 1
    input8 = torch.randn(2, 3, dtype=torch.complex64).numpy()
    input_dict8 = {
        "input": input8,
        "source": 0,
        "destination": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Float tensor, move axis 1 to -2
    input9 = torch.randn(2, 3, 4, 5).numpy()
    input_dict9 = {
        "input": input9,
        "source": 1,
        "destination": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    return list_of_inputs

generated_inputs = movedim_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('movedim', generated_inputs)
