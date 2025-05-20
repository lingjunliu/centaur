
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def matrix_exp_inputs():
    list_of_inputs = []

    # Input 1: Simple 2x2 float matrix
    A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3x3 matrix with negative values
    A = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4x4 complex matrix
    A = np.array([[1+1j, 0, 0, 0], [0, 2+2j, 0, 0], [0, 0, 3+3j, 0], [0, 0, 0, 4+4j]], dtype=np.complex64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Stacked matrices (batch of 2)
    A = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger matrix (5x5)
    A = np.random.rand(5, 5).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another complex matrix
    A = np.array([[1j, 1], [-1, -1j]], dtype=np.complex128)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2x2 matrix with high values
    A = np.array([[10.0, 5.0], [2.0, 8.0]], dtype=np.float32)
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
