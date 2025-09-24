
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def count_nonzero_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D integer tensor with some zeros
    input1 = np.array([0, 1, 2, 0, 3, 0], dtype=np.int64)
    input_dict1 = {"input": input1, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with negative values and zeros
    input2 = np.array([[-1.0, 0.0, 2.5], [0.0, -3.2, 0.0]], dtype=np.float32)
    input_dict2 = {"input": input2, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D boolean tensor
    input3 = np.array([[[True, False, True], [False, True, False]],
                       [[True, True, False], [False, False, True]]], dtype=np.bool_)
    input_dict3 = {"input": input3, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with complex numbers
    input4 = np.array([1 + 1j, 0 + 0j, 2 - 1j, 0 + 2j], dtype=np.complex64)
    input_dict4 = {"input": input4, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Empty tensor
    input5 = np.array([], dtype=np.int64)
    input_dict5 = {"input": input5, "dim": None}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = count_nonzero_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('countNonzero', generated_inputs)
