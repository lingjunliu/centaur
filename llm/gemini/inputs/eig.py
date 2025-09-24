
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def eig_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix with eigenvectors
    input1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_dict1 = {"input": input1, "eigenvectors": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Complex matrix
    input2 = np.array([[1 + 1j, 2], [3, 4 - 1j]], dtype=np.complex64)
    input_dict2 = {"input": input2, "eigenvectors": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Larger square matrix
    input3 = np.random.rand(5, 5).astype(np.float64)
    input_dict3 = {"input": input3, "eigenvectors": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Matrix with negative values
    input4 = np.array([[-1, 2], [-3, 4]], dtype=np.float32)
    input_dict4 = {"input": input4, "eigenvectors": True}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex matrix with larger values
    input5 = np.array([[10 + 5j, 20], [30, 40 - 10j]], dtype=np.complex64)
    input_dict5 = {"input": input5, "eigenvectors": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Float64 input
    input6 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    input_dict6 = {"input": input6, "eigenvectors": True}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Larger complex matrix with random values
    input7 = (np.random.rand(4, 4) + 1j * np.random.rand(4, 4)).astype(np.complex128)
    input_dict7 = {"input": input7, "eigenvectors": False}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = eig_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('eig', generated_inputs)
