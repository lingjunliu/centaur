
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 matrix with full rank
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict1 = {"input": torch.tensor(input1), "tol": 1e-8, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Singular 2x2 matrix
    input2 = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    input_dict2 = {"input": torch.tensor(input2), "tol": 1e-8, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3x3 matrix with rank 2
    input3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    input_dict3 = {"input": torch.tensor(input3), "tol": 1e-8, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1x1 matrix with zero value
    input4 = np.array([[0.0]], dtype=np.float32)
    input_dict4 = {"input": torch.tensor(input4), "tol": 1e-8, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1x1 matrix with non-zero value
    input5 = np.array([[5.0]], dtype=np.float32)
    input_dict5 = {"input": torch.tensor(input5), "tol": 1e-8, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Rectangular matrix (2x3) with full row rank
    input6 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict6 = {"input": torch.tensor(input6), "tol": 1e-8, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Rectangular matrix (3x2) with full column rank
    input7 = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
    input_dict7 = {"input": torch.tensor(input7), "tol": 1e-8, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Matrix with very small values
    input8 = np.array([[1e-9, 0.0], [0.0, 1e-9]], dtype=np.float32)
    input_dict8 = {"input": torch.tensor(input8), "tol": 1e-7, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Higher dimensional tensor (3x2x2) - treat as a stack of matrices
    input9 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]], dtype=np.float32)
    input_dict9 = {"input": torch.tensor(input9), "tol": 1e-8, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Hermitian matrix
    input10 = np.array([[1.0, 2.0 + 1j], [2.0 - 1j, 3.0]], dtype=np.complex64)
    input_dict10 = {"input": torch.tensor(input10), "tol": 1e-8, "rtol": 0.0, "hermitian": True}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Input 11: Matrix with complex values but not hermitian
    input11 = np.array([[1.0, 2.0 + 1j], [2.0 + 1j, 3.0]], dtype=np.complex64)
    input_dict11 = {"input": torch.tensor(input11), "tol": 1e-8, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Input 12: Zero matrix of size 3x3
    input12 = np.zeros((3, 3), dtype=np.float32)
    input_dict12 = {"input": torch.tensor(input12), "tol": 1e-8, "rtol": 0.0, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.matrix_rank_1"] = matrix_rank_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.linalg.matrix_rank_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.linalg.matrix_rank_1'.")

check_valid('torch.linalg.matrix_rank', generated_inputs['torch.linalg.matrix_rank_1'], lib="torch", suffix=1)
