
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def solve_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors, A is square
    A = np.random.rand(3, 3).astype(np.float32)
    B = np.random.rand(3, 1).astype(np.float32)
    input_dict = {"A": A, "B": B, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Double tensors, A is square
    A = np.random.rand(4, 4).astype(np.float64)
    B = np.random.rand(4, 3).astype(np.float64)
    input_dict = {"A": A, "B": B, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Complex tensors
    A = (np.random.rand(2, 2) + 1j * np.random.rand(2, 2)).astype(np.complex64)
    B = (np.random.rand(2, 1) + 1j * np.random.rand(2, 1)).astype(np.complex64)
    input_dict = {"A": A, "B": B, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Complex tensors, A is square
    A = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(np.complex128)
    B = (np.random.rand(3, 2) + 1j * np.random.rand(3, 2)).astype(np.complex128)
    input_dict = {"A": A, "B": B, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Float tensors, A is batch of square matrix
    A = np.random.rand(2, 3, 3).astype(np.float32)
    B = np.random.rand(2, 3, 1).astype(np.float32)
    input_dict = {"A": A, "B": B, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Float tensors, A is square, left=True
    A = np.random.rand(3, 3).astype(np.float32)
    B = np.random.rand(3, 1).astype(np.float32)
    input_dict = {"A": A, "B": B, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 9: Float tensors, A is square
    A = np.random.rand(3, 3).astype(np.float32)
    B = np.random.rand(3, 3).astype(np.float32)
    input_dict = {"A": A, "B": B, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 10: Basic float tensors, A is square, left = True, correct dimensions
    A = np.random.rand(3, 3).astype(np.float32)
    B = np.random.rand(3, 3).astype(np.float32)
    input_dict = {"A": A, "B": B, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 11: Float tensors, A is batch of square matrix, left = True
    A = np.random.rand(2, 3, 3).astype(np.float32)
    B = np.random.rand(2, 3, 3).astype(np.float32)
    input_dict = {"A": A, "B": B, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = solve_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('solve', generated_inputs)
