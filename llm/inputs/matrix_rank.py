
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def matrix_rank_inputs():
    list_of_inputs = []

    # Input 1: Simple float matrix
    A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "tol": 1e-8,
        "atol": 1e-8,
        "rtol": 1e-5,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float matrix with zero row
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "tol": 1e-8,
        "atol": 1e-8,
        "rtol": 1e-5,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex Hermitian matrix
    A = np.array([[1, 1j], [-1j, 2]], dtype=np.complex64)
    input_dict = {
        "A": A,
        "tol": 1e-8,
        "atol": 1e-8,
        "rtol": 1e-5,
        "hermitian": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float matrix with a small tolerance
    A = np.array([[1.0, 0.0001], [0.0001, 1.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "tol": 0.01,
        "atol": 1e-8,
        "rtol": 1e-5,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: A matrix with non-default atol and rtol
    A = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "tol": 1e-8,
        "atol": 0.1,
        "rtol": 0.1,
        "hermitian": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A matrix with negative values
    A = np.array([[1.0, -2.0], [-2.0, 4.0]], dtype=np.float32)
    input_dict = {
        "A": A,
        "tol": 1e-8,
        "atol": 1e-8,
        "rtol": 1e-5,
        "hermitian": False
    }
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
