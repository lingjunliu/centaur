
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def solve_ex_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[5.0], [11.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Complex tensors
    A = np.array([[1.0 + 1j, 2.0 - 1j], [3.0 + 0j, 4.0 + 2j]], dtype=np.complex64)
    B = np.array([[5.0 + 0j], [11.0 - 1j]], dtype=np.complex64)
    input_dict = {"A": A, "B": B, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shapes, solvable system
    A = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 2.0], [1.0, 0.0, 8.0]], dtype=np.float32)
    B = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Multiple right-hand sides
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[5.0, 6.0], [11.0, 12.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "check_errors": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values
    A = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    B = np.array([[5.0], [-11.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "check_errors": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = solve_ex_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('solve_ex', generated_inputs)
