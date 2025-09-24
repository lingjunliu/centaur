
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def fftn_inputs():
    list_of_inputs = []

    # Input 1: Basic example with complex input
    input1 = torch.randn(10, 10, dtype=torch.complex64).numpy()
    input_dict1 = {
        "input": input1,
        "s": None,
        "dim": None,
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Real input with specified dimensions and size
    input2 = torch.randn(5, 5, 5).numpy()
    input_dict2 = {
        "input": input2,
        "s": (10, 10, 5),
        "dim": (0, 1, 2),
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Integer input with 'ortho' normalization
    input3 = torch.randint(0, 10, (8, 8)).float().numpy()
    input_dict3 = {
        "input": input3,
        "s": None,
        "dim": None,
        "norm": "ortho"
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D input with padding and specific dimensions
    input4 = torch.randn(4, 4, 4).numpy()
    input_dict4 = {
        "input": input4,
        "s": (8, 4, 8),
        "dim": (0, 1, 2),
        "norm": "backward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex input with specified size and dim
    input5 = (torch.randn(3, 3) + 1j * torch.randn(3, 3)).numpy()
    input_dict5 = {
        "input": input5,
        "s": (6, 6),
        "dim": (0, 1),
        "norm": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float input with normalization
    input6 = torch.randn(2, 2).numpy()
    input_dict6 = {
        "input": input6,
        "s": None,
        "dim": None,
        "norm": "forward"
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = fftn_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fftn', generated_inputs)
