
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def lu_solve_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    LU_data = torch.randn(3, 3).numpy()
    LU_pivots = torch.arange(1, 4).numpy()
    b = torch.randn(3, 1).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of matrices
    LU_data = torch.randn(2, 4, 4).numpy()
    LU_pivots = torch.randint(1, 5, (2, 4)).numpy()
    b = torch.randn(2, 4, 2).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single Matrix, Long pivots
    LU_data = torch.randn(5, 5).numpy()
    LU_pivots = torch.arange(1, 6).long().numpy()
    b = torch.randn(5, 3).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different sizes
    LU_data = torch.randn(4, 4).numpy()
    LU_pivots = torch.arange(1, 5).numpy()
    b = torch.randn(4, 4).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative Values
    LU_data = torch.randn(3, 3).numpy() * -1
    LU_pivots = torch.arange(1, 4).numpy()
    b = torch.randn(3, 1).numpy() * -1
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = lu_solve_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('lu_solve', generated_inputs)
