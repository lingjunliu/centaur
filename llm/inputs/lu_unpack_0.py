
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def lu_unpack_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    LU_data = torch.randn(3, 3).numpy()
    LU_pivots = torch.randint(1, 3 + 1, (3,)).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2:  unpack_data = False
    LU_data = torch.randn(4, 4).numpy()
    LU_pivots = torch.randint(1, 4 + 1, (4,)).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": False,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3:  unpack_pivots = False
    LU_data = torch.randn(5, 5).numpy()
    LU_pivots = torch.randint(1, 5 + 1, (5,)).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Integer data
    LU_data = torch.randint(-10, 10, (3, 3)).numpy()
    LU_pivots = torch.randint(1, 3 + 1, (3,)).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch LU
    LU_data = torch.randn(2, 3, 3).numpy()
    LU_pivots = torch.randint(1, 3 + 1, (2, 3)).int().numpy()
    input_dict = {
        "LU_data": LU_data,
        "LU_pivots": LU_pivots,
        "unpack_data": True,
        "unpack_pivots": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.lu_unpack"] = lu_unpack_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.lu_unpack' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.lu_unpack'.")

check_valid('torch.lu_unpack', generated_inputs['torch.lu_unpack'], lib="torch")
