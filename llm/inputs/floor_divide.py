
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def floor_divide_inputs():
    list_of_inputs = []

    # Example 1: Basic integer division
    input1 = np.array([10, 20, 30]).astype(np.int32)
    other1 = np.array([3, 7, 2]).astype(np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: Floating point division
    input2 = np.array([10.5, 20.3, 30.9]).astype(np.float32)
    other2 = np.array([3.0, 7.0, 2.0]).astype(np.float32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: Negative numbers
    input3 = np.array([-10, -20, 30]).astype(np.int64)
    other3 = np.array([3, -7, 2]).astype(np.int64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: Multi-dimensional arrays
    input4 = np.array([[10, 20], [30, 40]]).astype(np.int32)
    other4 = np.array([[3, 7], [2, 5]]).astype(np.int32)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: Different shapes (broadcasting)
    input5 = np.array([[10, 20, 30], [40, 50, 60]]).astype(np.float64)
    other5 = np.array([2, 5, 10]).astype(np.float64)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Example 6: Scalar division
    input6 = np.array([10, 20, 30]).astype(np.int32)
    other6 = np.array(5).astype(np.int32)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Example 7: Zero division (shouldn't error, will produce inf)
    input7 = np.array([10, 20, 30]).astype(np.float32)
    other7 = np.array([0, 5, 0]).astype(np.float32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = floor_divide_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('floor_divide', generated_inputs)
