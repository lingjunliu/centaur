
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch
import numpy as np
import copy

def linalg_eigh_inputs():
    list_of_inputs = []

    # Input 1: Simple symmetric matrix (float64)
    A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    input_dict = {"A": A, "UPLO": 'L'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of symmetric matrices (float32)
    A = np.array([[[2.0, 1.0], [1.0, 3.0]], [[4.0, 2.0], [2.0, 5.0]]], dtype=np.float32)
    input_dict = {"A": A, "UPLO": 'U'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex Hermitian matrix (complex128)
    A = np.array([[2.0 + 0j, 1.0 - 1j], [1.0 + 1j, 3.0 + 0j]], dtype=np.complex128)
    input_dict = {"A": A, "UPLO": 'L'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of complex Hermitian matrices (complex64)
    A = np.array([[[2.0 + 0j, 1.0 - 1j], [1.0 + 1j, 3.0 + 0j]], [[4.0 + 0j, 2.0 - 2j], [2.0 + 2j, 5.0 + 0j]]], dtype=np.complex64)
    input_dict = {"A": A, "UPLO": 'U'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Larger symmetric matrix (float64)
    A = np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]], dtype=np.float64)
    input_dict = {"A": A, "UPLO": 'L'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Matrix with negative values (float32)
    A = np.array([[2.0, -1.0], [-1.0, 3.0]], dtype=np.float32)
    input_dict = {"A": A, "UPLO": 'U'}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: 3D batch of matrices
    A = np.random.rand(2, 3, 3)
    A = A + np.transpose(A, (0, 2, 1))
    input_dict = {"A": A.astype(np.float64), "UPLO": 'L'}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.linalg.eigh"] = linalg_eigh_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('eigh', generated_inputs['torch.linalg.eigh'], lib="torch")
