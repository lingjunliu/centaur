
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cholesky_ex_inputs():
    list_of_inputs = []

    # Input 1: Basic positive definite matrix
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float64)
    input_dict = {"input": A, "upper": False, "check_errors": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Upper triangular, check errors
    B = np.array([[1, 0], [0, 4]], dtype=np.float32)
    input_dict = {"input": B, "upper": True, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex matrix
    C = np.array([[4 + 0j, 1 + 1j], [1 - 1j, 2 + 0j]], dtype=np.complex128)
    input_dict = {"input": C, "upper": False, "check_errors": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched matrices
    D = np.array([[[4, 1], [1, 4]], [[9, 3], [3, 9]]], dtype=np.float64)
    input_dict = {"input": D, "upper": False, "check_errors": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Small matrix
    E = np.array([[10]], dtype=np.float32)
    input_dict = {"input": E, "upper": False, "check_errors": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = cholesky_ex_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cholesky_ex', generated_inputs)
