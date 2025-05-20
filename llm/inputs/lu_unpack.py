
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def lu_unpack_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float data and pivots
    LU_data = torch.randn(3, 3).numpy()
    LU_pivots = torch.arange(1, 4).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer data and pivots
    LU_data = torch.randint(1, 10, (4, 4)).numpy()
    LU_pivots = torch.arange(1, 5).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex data and pivots
    LU_data = (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy()
    LU_pivots = torch.arange(1, 3).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D data, 1D pivots, no unpacking of pivots
    LU_data = torch.randn(5, 5).numpy()
    LU_pivots = torch.arange(1, 6).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: 2D data, 1D pivots, no unpacking of data
    LU_data = torch.randn(5, 5).numpy()
    LU_pivots = torch.arange(1, 6).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": False,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = lu_unpack_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lu_unpack', generated_inputs)
