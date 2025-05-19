
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def matrix_rank_inputs():
    list_of_inputs = []

    A = np.array([[1, 0], [0, 1]])
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1, 2], [2, 4]])
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    input_dict = {"A": A, "tol": 1e-8, "atol": 1e-8, "rtol": 1e-5, "hermitian": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = matrix_rank_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('matrix_rank', generated_inputs)
