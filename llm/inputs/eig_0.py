
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_linalg_eig_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 real matrix
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 complex matrix
    A = np.array([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j], [7+7j, 8+8j, 9+9j]], dtype=np.complex128)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of 2 2x2 real matrices
    A = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4x4 real matrix with negative values
    A = np.array([[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0], [9.0, -10.0, 11.0, -12.0], [-13.0, 14.0, -15.0, 16.0]], dtype=np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.eig"] = torch_linalg_eig_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.eig' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.eig'.")

check_valid('torch.linalg.eig', generated_inputs['torch.linalg.eig'], lib="torch")
