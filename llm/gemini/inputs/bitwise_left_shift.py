
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def bitwise_left_shift_inputs():
    list_of_inputs = []

    # Input 1: Basic case with positive integers
    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    other1 = np.array([1, 2, 0, 3], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Broadcasting with a scalar
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other2 = np.array(2, dtype=np.int64)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional arrays
    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    other3 = np.array([[[0, 1], [2, 0]], [[1, 0], [0, 2]]], dtype=np.int32)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different integer types
    input4 = np.array([1, 2, 3], dtype=np.int16)
    other4 = np.array([1, 2, 1], dtype=np.int16)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger shift values
    input5 = np.array([1, 2, 3], dtype=np.int32)
    other5 = np.array([10, 5, 2], dtype=np.int32)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = bitwise_left_shift_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bitwise_left_shift', generated_inputs)
