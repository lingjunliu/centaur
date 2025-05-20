
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def linalg_eigvals_inputs():
    list_of_inputs = []

    A = np.random.rand(2, 2).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(3, 3).astype(np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    A = A.astype(np.complex64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    A = A.astype(np.complex128)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.randn(4, 4).astype(np.float32)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(2, 3, 3).astype(np.float64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(2, 2, 2) + 1j * np.random.rand(2, 2, 2)
    A = A.astype(np.complex64)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(4, 5, 5) + 1j * np.random.rand(4, 5, 5)
    A = A.astype(np.complex128)
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(2, 2).astype(np.float32) * -1
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    A = np.random.rand(3, 3).astype(np.float64) * -1
    input_dict = {"A": A}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = linalg_eigvals_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('eigvals', generated_inputs)
