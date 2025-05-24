
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def maximum_inputs():
    list_of_inputs = []

    # Case 1: Basic integer tensors
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([3, 1, 4], dtype=np.int32)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 2: Basic float tensors
    input1 = np.array([1.0, 2.5, 3.2], dtype=np.float32)
    input2 = np.array([3.1, 1.2, 4.0], dtype=np.float32)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 3: Negative values
    input1 = np.array([-1, 2, -3], dtype=np.int32)
    input2 = np.array([3, -1, 4], dtype=np.int32)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 4: Different shapes (but broadcastable)
    input1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input2 = np.array([0, 5], dtype=np.int32)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 5: Multi-dimensional tensors
    input1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input2 = np.array([[[0, 3], [2, 5]], [[4, 7], [6, 9]]], dtype=np.float32)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 6: Mixed dtypes (int and float) - should be promoted to float
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([3.0, 1.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 7: Zero-dimensional tensors
    input1 = np.array(5, dtype=np.int32)
    input2 = np.array(10, dtype=np.int32)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 8: More complex broadcastable shapes
    input1 = np.random.rand(2, 3, 4).astype(np.float32)
    input2 = np.random.rand(3, 1).astype(np.float32)  # Broadcastable to (2, 3, 4)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 9: One element tensors
    input1 = np.array([10], dtype=np.int32)
    input2 = np.array([5], dtype=np.int32)
    list_of_inputs.append({"input": input1, "other": input2})

    return list_of_inputs

generated_inputs = maximum_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('maximum', generated_inputs)
