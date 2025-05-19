
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def cholesky_solve_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    A = np.random.rand(3, 3)
    A = A @ A.T  # Make A positive definite
    B = np.random.rand(3, 2)
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape, float type
    A = np.random.rand(5, 5)
    A = A @ A.T
    B = np.random.rand(5, 1)
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Double type
    A = np.random.rand(4, 4).astype(np.float64)
    A = A @ A.T
    B = np.random.rand(4, 3).astype(np.float64)
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different B shape and A is complex
    A = np.random.rand(2, 2) + 1j*np.random.rand(2, 2)
    A = A @ A.conj().T
    B = np.random.rand(2, 4) + 1j*np.random.rand(2, 4)
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: A and B are complex with more dims
    A = (np.random.rand(2, 3, 3) + 1j*np.random.rand(2, 3, 3))
    A = np.einsum('aij,ajk->aik', A, A.conj().transpose(0, 2, 1))
    B = (np.random.rand(2, 3, 4) + 1j*np.random.rand(2, 3, 4))
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger matrices
    A = np.random.rand(10, 10)
    A = A @ A.T
    B = np.random.rand(10, 5)
    input_dict = {"A": A, "B": B}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = cholesky_solve_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cholesky_solve', generated_inputs)
