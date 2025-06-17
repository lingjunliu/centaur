
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_inv_inputs():
    list_of_inputs = []

    # Input 1: Simple float matrix
    A = np.random.rand(3, 3).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Double matrix
    A = np.random.rand(2, 2).astype(np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex matrix
    A = (np.random.rand(4, 4) + 1j * np.random.rand(4, 4)).astype(np.complex128)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of matrices
    A = np.random.rand(2, 3, 3).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Higher dimensional batch of matrices
    A = np.random.rand(2, 2, 2, 2).astype(np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.linalg.inv"] = linalg_inv_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.inv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.inv'.")

check_valid('torch.linalg.inv', generated_inputs['torch.linalg.inv'], lib="torch")
