
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def less_equal_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0])
    other1 = np.array([2.0, 2.0, 1.0])
    list_of_inputs.append({"input": input1, "other": other1})

    # Test case 2: Integer tensors
    input2 = np.array([1, 2, 3], dtype=np.int32)
    other2 = np.array([2, 1, 3], dtype=np.int32)
    list_of_inputs.append({"input": input2, "other": other2})

    # Test case 3: Mixed types (float and int)
    input3 = np.array([1.0, 2.0, 3.0])
    other3 = np.array([2, 1, 3], dtype=np.int32)
    list_of_inputs.append({"input": input3, "other": other3})

    # Test case 4: Negative values
    input4 = np.array([-1.0, -2.0, 3.0])
    other4 = np.array([-2.0, 0.0, 3.0])
    list_of_inputs.append({"input": input4, "other": other4})

    # Test case 5: Multidimensional arrays
    input5 = np.array([[1, 2], [3, 4]])
    other5 = np.array([[2, 1], [4, 3]])
    list_of_inputs.append({"input": input5, "other": other5})

    # Test case 6: Scalar
    input6 = np.array([1, 2, 3])
    other6 = np.array(2)
    list_of_inputs.append({"input": input6, "other": other6})

    # Test case 7: Different shapes (broadcasting) - numpy automatically broadcasts
    input7 = np.array([[1, 2, 3], [4, 5, 6]])
    other7 = np.array([2, 4, 5])
    list_of_inputs.append({"input": input7, "other": other7})

    # Test case 8: Zero values
    input8 = np.array([0.0, 2.0, 0.0])
    other8 = np.array([0.0, 1.0, 1.0])
    list_of_inputs.append({"input": input8, "other": other8})
    
    return list_of_inputs

generated_inputs = less_equal_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('less_equal', generated_inputs)
