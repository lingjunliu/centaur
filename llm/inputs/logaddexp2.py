
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def logaddexp2_inputs():
    generated_inputs = []

    # Test case 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other1 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    generated_inputs.append({"input": input1, "other": other1})

    # Test case 2: Negative values
    input2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    other2 = np.array([-4.0, -5.0, -6.0], dtype=np.float32)
    generated_inputs.append({"input": input2, "other": other2})

    # Test case 3: Zero values
    input3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    other3 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    generated_inputs.append({"input": input3, "other": other3})

    # Test case 4: Different shapes (but broadcastable)
    input4 = np.array([1.0, 2.0], dtype=np.float32)
    other4 = np.array(3.0, dtype=np.float32)
    generated_inputs.append({"input": input4, "other": other4})

    # Test case 5: Multi-dimensional tensors
    input5 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    other5 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    generated_inputs.append({"input": input5, "other": other5})

    # Test case 9: Different data types (float64)
    input9 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    other9 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    generated_inputs.append({"input": input9, "other": other9})

    return generated_inputs

generated_inputs = logaddexp2_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logaddexp2', generated_inputs)
