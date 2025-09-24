
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def matrix_exp_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 matrix
    A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 matrix with negative values
    A = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex 2x2 matrix
    A = np.array([[1.0 + 1j, 0.0], [0.0, 1.0 - 1j]], dtype=np.complex64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: 4x4 matrix with some larger values
    A = np.array([[2.0, 1.0, 0.0, 0.0], [0.0, 2.0, 1.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 2.0]], dtype=np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: A batch of matrices (3x2x2)
    A = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 1.0]]], dtype=np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger random matrix
    A = np.random.rand(5, 5).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = matrix_exp_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('matrix_exp', generated_inputs)
