
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def eq_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0])
    input2 = np.array([1.0, 2.0, 4.0])
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 2: Integer tensors
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 3: Different shapes (broadcastable)
    input1 = np.array([[1, 2], [3, 4]])
    input2 = np.array([1, 2])
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 4: Negative values and different dtypes
    input1 = np.array([-1.0, 0.0, 1.0])
    input2 = np.array([-1, 0, 1], dtype=np.int64)
    list_of_inputs.append({"input": input1, "other": input2})
    
    # Case 5: Multi-dimensional array
    input1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    input2 = np.array([[[1, 2], [3, 4]], [[5, 7], [7, 8]]])
    list_of_inputs.append({"input": input1, "other": input2})
    
    # Case 6: Zero-dimensional array
    input1 = np.array(5)
    input2 = np.array(5)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 7: Complex numbers
    input1 = np.array([1 + 1j, 2 + 2j])
    input2 = np.array([1 + 1j, 3 + 3j])
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 8: Boolean arrays
    input1 = np.array([True, False, True])
    input2 = np.array([True, True, False])
    list_of_inputs.append({"input": input1, "other": input2})

    return list_of_inputs

generated_inputs = eq_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('eq', generated_inputs)
