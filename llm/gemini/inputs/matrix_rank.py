
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor
    A = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor, hermitian=True
    A = np.array([[1+0j, 2-1j], [2+1j, 3+0j]], dtype=np.complex64)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different tolerance
    A = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 0.5, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rectangular matrix
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zero matrix
    A = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Matrix with negative values
    A = np.array([[1.0, -2.0], [-2.0, 4.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Single element matrix
    A = np.array([[5.0]], dtype=np.float32)
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = matrix_rank_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('matrix_rank', generated_inputs)
