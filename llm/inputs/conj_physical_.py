
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def conj_physical__inputs():
    list_of_inputs = []

    # Input 1: Complex float64 tensor
    input1 = np.array([[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]], dtype=np.complex128)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Complex float32 tensor with different values
    input2 = np.array([[-1 - 2j, 3 + 4j], [-5 - 6j, 7 + 8j]], dtype=np.complex64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D Complex float64 tensor
    input3 = np.array([[[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]],
                       [[9 - 10j, 11 + 12j], [13 - 14j, 15 + 16j]]], dtype=np.complex128)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D Complex float32 tensor
    input4 = np.array([1j, 2 - 1j, 3 + 2j, -4 - 3j], dtype=np.complex64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Complex float64 tensor with zero values
    input5 = np.array([[0 + 0j, 0 - 0j], [0 + 0j, 0 - 0j]], dtype=np.complex128)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 4D complex float32 tensor
    input6 = np.random.rand(2, 2, 2, 2).astype(np.float32) + 1j * np.random.rand(2, 2, 2, 2).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = conj_physical__inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('conj_physical_', generated_inputs)
