
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cholesky_ex_inputs():
    list_of_inputs = []

    # Input 1: Basic positive definite matrix (float32)
    A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float32)
    input_dict = {"input": A, "upper": False, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data type (float64), upper=True
    A = np.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]], dtype=np.float64)
    input_dict = {"input": A, "upper": True, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Batch of matrices (3D tensor)
    A = np.array([[[4, 12, -16], [12, 37, -43], [-16, -43, 98]],
                  [[9, -6, 0], [-6, 5, 0], [0, 0, 1]]], dtype=np.float32)
    input_dict = {"input": A, "upper": False, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex input
    A = np.array([[4+0j, 1+1j], [1-1j, 2+0j]], dtype=np.complex64)
    input_dict = {"input": A, "upper": False, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Another positive definite matrix
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    input_dict = {"input": A, "upper": False, "check_errors": True}
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
