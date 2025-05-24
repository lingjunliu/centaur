
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def minimum_inputs():
    list_of_inputs = []

    # Input 1: Basic integer tensors
    input1 = np.array([1, 2, 3], dtype=np.int32)
    other1 = np.array([3, 1, 2], dtype=np.int32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensors with negative values
    input2 = np.array([-1.5, 2.0, -3.7], dtype=np.float32)
    other2 = np.array([0.5, -1.0, 1.2], dtype=np.float32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Tensors with different shapes (but broadcastable)
    input3 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    other3 = np.array([0, 5], dtype=np.int64)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Higher dimensional tensors
    input4 = np.random.rand(2, 3, 4).astype(np.float64)
    other4 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Tensors with some NaN values
    input5 = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    other5 = np.array([2.0, 4.0, np.nan], dtype=np.float32)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Boolean tensors
    input6 = np.array([True, False, True], dtype=bool)
    other6 = np.array([False, True, False], dtype=bool)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Scalar inputs
    input7 = np.array(5, dtype=np.int32)
    other7 = np.array(10, dtype=np.int32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Input 8: Mixed dtypes (int and float)
    input8 = np.array([1, 2, 3], dtype=np.int32)
    other8 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = minimum_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('minimum', generated_inputs)
