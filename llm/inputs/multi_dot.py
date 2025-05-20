
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def multi_dot_inputs():
    list_of_inputs = []

    # Test case 1: Basic 2D matrices
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    input_dict = {'tensors': [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Chain of 3 matrices
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    c = np.random.rand(4, 2).astype(np.float32)
    input_dict = {'tensors': [a, b, c]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Matrices with negative values
    a = (np.random.rand(2, 3) - 0.5).astype(np.float32)
    b = (np.random.rand(3, 4) - 0.5).astype(np.float32)
    input_dict = {'tensors': [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 4: A larger chain of matrices
    a = np.random.rand(5, 3).astype(np.float32)
    b = np.random.rand(3, 2).astype(np.float32)
    c = np.random.rand(2, 7).astype(np.float32)
    d = np.random.rand(7, 2).astype(np.float32)
    e = np.random.rand(2, 4).astype(np.float32)
    input_dict = {'tensors': [a, b, c, d, e]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Matrices with float64
    a = np.random.rand(2, 3).astype(np.float64)
    b = np.random.rand(3, 4).astype(np.float64)
    input_dict = {'tensors': [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = multi_dot_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('multi_dot', generated_inputs)
