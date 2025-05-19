
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def solve_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors, left=True
    A = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensors, left=False
    A = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different shapes, left=True
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    B = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "left": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different shapes, left=False
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    B = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]], dtype=np.float32)
    input_dict = {"A": A, "B": B, "left": False}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Negative values, left=True
    A = np.array([[-1.0, 2.0], [3.0, -5.0]], dtype=np.float32)
    B = np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float32)
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
