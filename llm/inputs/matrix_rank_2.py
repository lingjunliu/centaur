
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def matrix_rank_inputs():
    list_of_inputs = []

    # Example 1: Basic 2D matrix with full rank
    input1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    tol1 = np.array([1e-8], dtype=np.float32)
    input_dict1 = {"input": input1, "tol": tol1, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Matrix with rank deficiency
    input2 = np.array([[1, 2], [2, 4]], dtype=np.float32)
    tol2 = np.array([1e-8], dtype=np.float32)
    input_dict2 = {"input": input2, "tol": tol2, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: 3D tensor
    input3 = np.random.rand(2, 3, 4).astype(np.float32)
    tol3 = np.array([1e-8], dtype=np.float32)
    input_dict3 = {"input": input3, "tol": tol3, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Complex matrix
    input4 = np.array([[1+1j, 2], [3, 4-1j]], dtype=np.complex64)
    tol4 = np.array([1e-8], dtype=np.float32)
    input_dict4 = {"input": input4, "tol": tol4, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["torch.linalg.matrix_rank_2"] = matrix_rank_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_rank_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_rank_2'.")

check_valid('torch.linalg.matrix_rank', generated_inputs['torch.linalg.matrix_rank_2'], lib="torch")
