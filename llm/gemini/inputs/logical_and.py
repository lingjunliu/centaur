
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def logical_and_inputs():
    list_of_inputs = []

    # Case 1: Basic boolean tensors
    input1 = np.array([True, False, True, False])
    input2 = np.array([True, True, False, False])
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 2: Integer tensors (0 is False, non-zero is True)
    input1 = np.array([1, 0, 2, 0])
    input2 = np.array([0, 1, 0, 3])
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 3: Floating-point tensors (0.0 is False, non-zero is True)
    input1 = np.array([1.0, 0.0, -2.5, 0.0])
    input2 = np.array([0.0, 1.5, 0.0, -3.0])
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 4: Multi-dimensional tensors
    input1 = np.array([[True, False], [True, True]])
    input2 = np.array([[False, True], [True, False]])
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 5: Different shapes (but broadcastable)
    input1 = np.array([[True, False], [True, True]])
    input2 = np.array([True, False])  # Broadcastable to (2, 2)
    list_of_inputs.append({"input": input1, "other": input2})

    # Case 6: Negative numbers in integer arrays
    input1 = np.array([-1, 0, 1, -2])
    input2 = np.array([2, -3, 0, 1])
    list_of_inputs.append({"input": input1, "other": input2})
    
    # Case 7: More complex shapes
    input1 = np.random.choice([True, False], size=(2, 3, 4))
    input2 = np.random.choice([True, False], size=(2, 3, 4))
    list_of_inputs.append({"input": input1, "other": input2})

    return list_of_inputs

generated_inputs = logical_and_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logical_and', generated_inputs)
