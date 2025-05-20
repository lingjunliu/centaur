
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cross_inputs():
    list_of_inputs = []

    # Example 1: Basic 2D tensors with dim=1
    a = torch.randn(4, 3).numpy()
    b = torch.randn(4, 3).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Different tensor sizes with dim=0
    a = torch.randn(3, 4).numpy()
    b = torch.randn(3, 4).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 3D tensors with explicit dim=2
    a = torch.randn(2, 3, 3).numpy()
    b = torch.randn(2, 3, 3).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4: Integer tensors with default dim
    a = torch.randint(-5, 5, (4, 3)).numpy()
    b = torch.randint(-5, 5, (4, 3)).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Float tensors with negative values
    a = (torch.randn(4, 3) * -1).numpy()
    b = (torch.randn(4, 3) * -1).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: 1D tensors, should cause error as dimension must be > 1 and size = 3 at dim=-1/0
    a = torch.randn(3).numpy()
    b = torch.randn(3).numpy()
    input_dict = {"input": a, "other": b}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = cross_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cross', generated_inputs)
