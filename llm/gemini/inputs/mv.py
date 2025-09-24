
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def mv_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    input_matrix = np.random.randn(5, 3).astype(np.float32)
    input_vector = np.random.randn(3).astype(np.float32)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensors
    input_matrix = np.random.randint(1, 10, size=(4, 5)).astype(np.int32)
    input_vector = np.random.randint(1, 10, size=(5)).astype(np.int32)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative values
    input_matrix = np.random.randn(6, 4).astype(np.float32) * -1
    input_vector = np.random.randn(4).astype(np.float32) * -1
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes
    input_matrix = np.random.randn(2, 7).astype(np.float32)
    input_vector = np.random.randn(7).astype(np.float32)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Double precision
    input_matrix = np.random.randn(3, 2).astype(np.float64)
    input_vector = np.random.randn(2).astype(np.float64)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex tensors
    input_matrix = (np.random.randn(4, 3) + 1j * np.random.randn(4, 3)).astype(np.complex64)
    input_vector = (np.random.randn(3) + 1j * np.random.randn(3)).astype(np.complex64)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Matrix with a single row
    input_matrix = np.random.randn(1, 5).astype(np.float32)
    input_vector = np.random.randn(5).astype(np.float32)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Matrix with a single column
    input_matrix = np.random.randn(5, 1).astype(np.float32)
    input_vector = np.random.randn(1).astype(np.float32)
    input_dict = {"input": input_matrix, "vec": input_vector}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = mv_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('mv', generated_inputs)
