
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_tensorinv_inputs():
    list_of_inputs = []

    # Test case 1: Basic square matrix
    A = np.eye(4)
    input_dict = {"A": A, "ind": 1, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Reshaped matrix to satisfy the constraint
    A = np.eye(6).reshape(2,3,1,1)
    input_dict = {"A": A, "ind": 2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Different ind value
    A = np.eye(4).reshape(2,2,1,1)
    input_dict = {"A": A, "ind": 2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 4: Float tensor
    A = np.eye(4, dtype=np.float64).reshape(2,2,1,1)
    input_dict = {"A": A, "ind": 2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 5: Complex tensor
    A = np.eye(4, dtype=np.complex128).reshape(2,2,1,1)
    input_dict = {"A": A, "ind": 2, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.linalg.tensorinv"] = linalg_tensorinv_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.tensorinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.tensorinv'.")

check_valid('torch.linalg.tensorinv', generated_inputs['torch.linalg.tensorinv'], lib="torch")
