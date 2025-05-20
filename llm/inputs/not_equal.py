
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def not_equal_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensors
    input1 = np.array([1, 2, 3, 4])
    input2 = np.array([1, 3, 2, 4])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Float tensors with different shapes
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([1.0, 3.0])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative numbers and different dtypes
    input1 = np.array([-1, -2, 0, 1], dtype=np.int64)
    input2 = np.array([0, -2, 1, 2], dtype=np.int32)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Multidimensional tensors
    input1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input2 = np.array([[[1, 3], [3, 5]], [[5, 7], [7, 9]]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Zero-dimensional tensors (scalars)
    input1 = np.array(5)
    input2 = np.array(6)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Complex numbers
    input1 = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    input2 = np.array([1 + 1j, 3 + 2j, 4 + 3j])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 7: Boolean tensors
    input1 = np.array([True, False, True])
    input2 = np.array([False, False, True])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = not_equal_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('not_equal', generated_inputs)
