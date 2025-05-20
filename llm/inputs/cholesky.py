
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cholesky_inputs():
    list_of_inputs = []

    # Input 1: Basic symmetric positive-definite matrix
    a = np.random.rand(3, 3)
    a = a @ a.T + np.eye(3) * 1e-3
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of symmetric positive-definite matrices
    a = np.random.rand(2, 2, 2)
    a = a @ np.transpose(a, (0, 2, 1)) + np.eye(2) * 1e-3
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger symmetric positive-definite matrix
    a = np.random.rand(5, 5)
    a = a @ a.T + np.eye(5) * 1e-3
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch of larger matrices
    a = np.random.rand(3, 4, 4)
    a = a @ np.transpose(a, (0, 2, 1)) + np.eye(4) * 1e-3
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Another valid case
    a = np.random.rand(2, 3, 3)
    a = a @ np.transpose(a, (0, 2, 1)) + np.eye(3) * 1e-3
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element matrix
    a = np.array([[1.0]])
    input_dict = {"input": a}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = cholesky_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cholesky', generated_inputs)
