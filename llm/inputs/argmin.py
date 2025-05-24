
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def argmin_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D float tensor, no dim specified
    input_tensor = torch.randn(4, 4).numpy()
    input_dict = {"input": input_tensor, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 2D int tensor, dim=1
    input_tensor = torch.randint(-5, 5, (3, 5)).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 3D float tensor, dim=0, keepdim=True
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 1D float tensor, dim=0 (should be equivalent to None), keepdim=False
    input_tensor = torch.randn(7).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 2D float tensor with negative values, dim=1, keepdim=True
    input_tensor = (torch.randn(3, 5) - 2).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: 4D float tensor, dim=2
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {"input": input_tensor, "dim": 2, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = argmin_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('argmin', generated_inputs)
