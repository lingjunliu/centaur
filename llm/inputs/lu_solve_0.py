
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def lu_solve_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    LU_data = torch.randn(3, 3).numpy()
    LU_pivots = torch.arange(1, 4).int().numpy()
    b = torch.randn(3, 1).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different shaped b
    LU_data = torch.randn(4, 4).numpy()
    LU_pivots = torch.arange(1, 5).int().numpy()
    b = torch.randn(4, 2).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer tensors
    LU_data = torch.randn(2, 2).numpy()
    LU_pivots = torch.arange(1, 3).int().numpy()
    b = torch.randn(2, 1).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Negative values
    LU_data = torch.randn(5, 5).numpy()
    LU_pivots = torch.arange(1, 6).int().numpy()
    b = torch.randn(5, 3).numpy()
    b[0][0] = -1
    LU_data[0][0] = -1
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Batched b
    LU_data = torch.randn(2, 3, 3).numpy()
    LU_pivots = torch.randint(1,4,(2,3)).int().numpy()
    b = torch.randn(2, 3, 1).numpy()
    input_dict = {"b": b, "LU_data": LU_data, "LU_pivots": LU_pivots}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.lu_solve"] = lu_solve_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.lu_solve' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lu_solve'.")

check_valid('torch.lu_solve', generated_inputs['torch.lu_solve'], lib="torch")
