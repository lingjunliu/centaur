
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def lu_solve_inputs():
    list_of_inputs = []

    # Input 1
    A = torch.randn(3, 3)
    b = torch.randn(3, 1)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b_np = b.numpy()
    LU_data_np = LU_data.numpy()
    LU_pivots_np = LU_pivots.numpy()
    input_dict = {
        "b": b_np,
        "LU_data": LU_data_np,
        "LU_pivots": LU_pivots_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    A = torch.randn(4, 4)
    b = torch.randn(4, 2)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b_np = b.numpy()
    LU_data_np = LU_data.numpy()
    LU_pivots_np = LU_pivots.numpy()
    input_dict = {
        "b": b_np,
        "LU_data": LU_data_np,
        "LU_pivots": LU_pivots_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    A = torch.randn(5, 5)
    b = torch.randn(5, 1)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b_np = b.numpy()
    LU_data_np = LU_data.numpy()
    LU_pivots_np = LU_pivots.numpy()
    input_dict = {
        "b": b_np,
        "LU_data": LU_data_np,
        "LU_pivots": LU_pivots_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    A = torch.randn(2, 2)
    b = torch.randn(2, 3)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b_np = b.numpy()
    LU_data_np = LU_data.numpy()
    LU_pivots_np = LU_pivots.numpy()
    input_dict = {
        "b": b_np,
        "LU_data": LU_data_np,
        "LU_pivots": LU_pivots_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    A = torch.randn(6, 6)
    b = torch.randn(6, 2)
    LU_data, LU_pivots = torch.linalg.lu_factor(A)
    b_np = b.numpy()
    LU_data_np = LU_data.numpy()
    LU_pivots_np = LU_pivots.numpy()
    input_dict = {
        "b": b_np,
        "LU_data": LU_data_np,
        "LU_pivots": LU_pivots_np
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = lu_solve_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lu_solve', list_of_inputs)
