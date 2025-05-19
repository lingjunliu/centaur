
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def bitwise_or_inputs():
    generated_inputs = []

    # Test case 1: Basic integer tensors
    input1 = np.array([1, 2, 3, 4], dtype=np.int32)
    other1 = np.array([4, 3, 2, 1], dtype=np.int32)
    generated_inputs.append({"input": input1, "other": other1})

    # Test case 2: Different shapes, but broadcastable
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other2 = np.array([1, 0], dtype=np.int64)
    generated_inputs.append({"input": input2, "other": other2})

    # Test case 3: Scalar value
    input3 = np.array([5, 6, 7, 8], dtype=np.int8)
    other3 = np.array(3, dtype=np.int8)
    generated_inputs.append({"input": input3, "other": other3})

    # Test case 4: Multi-dimensional arrays
    input4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
    other4 = np.array([[[8, 7], [6, 5]], [[4, 3], [2, 1]]], dtype=np.uint8)
    generated_inputs.append({"input": input4, "other": other4})
    
    # Test case 5: Boolean arrays, which should also work (implicitly cast to integers)
    input5 = np.array([True, False, True, False], dtype=bool)
    other5 = np.array([False, True, False, True], dtype=bool)
    generated_inputs.append({"input": input5, "other": other5})
    
    # Test case 6: Negative integers
    input6 = np.array([-1, -2, -3, -4], dtype=np.int32)
    other6 = np.array([4, 3, 2, 1], dtype=np.int32)
    generated_inputs.append({"input": input6, "other": other6})

    # Test case 7: Mixed positive and negative integers
    input7 = np.array([-1, 2, -3, 4], dtype=np.int64)
    other7 = np.array([1, -2, 3, -4], dtype=np.int64)
    generated_inputs.append({"input": input7, "other": other7})
    
    # Test case 8: uint8 array
    input8 = np.array([255, 128, 64, 32], dtype=np.uint8)
    other8 = np.array([1, 2, 4, 8], dtype=np.uint8)
    generated_inputs.append({"input": input8, "other": other8})
    
    return generated_inputs

generated_inputs = bitwise_or_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('bitwise_or', generated_inputs)
