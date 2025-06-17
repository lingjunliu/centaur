
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_1 = np.array([[1.0, 2.0], [2.0, 4.0]])
    tol_1 = np.array([1e-8])
    input_dict_1 = {"input": input_1, "tol": tol_1, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Integer tensor
    input_2 = np.array([[1, 0, 1], [2, -1, 0], [3, -1, 1]])
    tol_2 = np.array([1e-8])
    input_dict_2 = {"input": input_2, "tol": tol_2, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Complex tensor, hermitian=True
    input_3 = np.array([[1 + 0j, 2 - 1j], [2 + 1j, 5 + 0j]])
    tol_3 = np.array([1e-8])
    input_dict_3 = {"input": input_3, "tol": tol_3, "hermitian": True}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Tensor with a small value as tolerance
    input_4 = np.array([[1.0, 0.0], [0.0, 0.0]])
    tol_4 = np.array([0.5])
    input_dict_4 = {"input": input_4, "tol": tol_4, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: 3D tensor
    input_5 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    tol_5 = np.array([1e-8])
    input_dict_5 = {"input": input_5, "tol": tol_5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict_5))
    
    # Input 6: Float64 tensor
    input_6 = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    tol_6 = np.array([1e-8])
    input_dict_6 = {"input": input_6, "tol": tol_6, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    return list_of_inputs

generated_inputs["torch.linalg.matrix_rank_2"] = matrix_rank_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_rank_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_rank_2'.")

check_valid('torch.linalg.matrix_rank', generated_inputs['torch.linalg.matrix_rank_2'], lib="torch")
