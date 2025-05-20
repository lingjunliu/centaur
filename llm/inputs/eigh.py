
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def torch_linalg_eigh_inputs():
    generated_inputs = []

    # Input 1: Real symmetric matrix, UPLO='L' (default)
    A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    input_dict = {"A": A, "UPLO": 'L'}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Real symmetric matrix, UPLO='U'
    A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    input_dict = {"A": A, "UPLO": 'U'}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex Hermitian matrix, UPLO='L'
    A = np.array([[2.0 + 0.0j, 1.0 - 1.0j], [1.0 + 1.0j, 3.0 + 0.0j]], dtype=np.complex128)
    input_dict = {"A": A, "UPLO": 'L'}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex Hermitian matrix, UPLO='U'
    A = np.array([[2.0 + 0.0j, 1.0 - 1.0j], [1.0 + 1.0j, 3.0 + 0.0j]], dtype=np.complex64)
    input_dict = {"A": A, "UPLO": 'U'}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch of real symmetric matrices
    A = np.array([[[2.0, 1.0], [1.0, 3.0]], [[4.0, 2.0], [2.0, 5.0]]], dtype=np.float64)
    input_dict = {"A": A, "UPLO": 'L'}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger real symmetric matrix
    A = np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]], dtype=np.float64)
    input_dict = {"A": A, "UPLO": 'L'}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Batch of complex hermitian matrices with different UPLO
    A = np.array([[[2.0 + 0.0j, 1.0 - 1.0j], [1.0 + 1.0j, 3.0 + 0.0j]], [[4.0 + 0.0j, 2.0 - 2.0j], [2.0 + 2.0j, 5.0 + 0.0j]]], dtype=np.complex128)
    input_dict = {"A": A, "UPLO": 'U'}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    return generated_inputs

generated_inputs = torch_linalg_eigh_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('eigh', generated_inputs)
