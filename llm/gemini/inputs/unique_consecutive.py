
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def unique_consecutive_inputs():
    list_of_inputs = []

    # Example 1: 1D int tensor
    input1 = np.array([1, 1, 2, 2, 3, 1, 1, 2], dtype=np.int64)
    input_dict1 = {
        "input": input1,
        "dim": None,
        "return_counts": False,
        "return_inverse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: 1D float tensor with return_counts
    input2 = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 1.0, 2.0], dtype=np.float32)
    input_dict2 = {
        "input": input2,
        "dim": None,
        "return_counts": True,
        "return_inverse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Example 3: 2D int tensor with dim specified
    input3 = np.array([[1, 1, 2], [2, 3, 1], [1, 2, 2]], dtype=np.int32)
    input_dict3 = {
        "input": input3,
        "dim": 0,
        "return_counts": False,
        "return_inverse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Example 4: 2D float tensor with dim and return_inverse
    input4 = np.array([[1.0, 1.0, 2.0], [2.0, 3.0, 1.0], [1.0, 2.0, 2.0]], dtype=np.float64)
    input_dict4 = {
        "input": input4,
        "dim": 1,
        "return_counts": False,
        "return_inverse": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Example 5: 3D int tensor with all options
    input5 = np.array([[[1, 1, 2], [2, 3, 1]], [[1, 2, 2], [3, 3, 4]]], dtype=np.int16)
    input_dict5 = {
        "input": input5,
        "dim": 2,
        "return_counts": True,
        "return_inverse": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Example 6: 1D tensor with negative values
    input6 = np.array([-1, -1, 0, 1, 1, 2, -1, -1], dtype=np.int64)
    input_dict6 = {
        "input": input6,
        "dim": None,
        "return_counts": False,
        "return_inverse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Example 7: Empty tensor
    input7 = np.array([], dtype=np.int64)
    input_dict7 = {
        "input": input7,
        "dim": None,
        "return_counts": False,
        "return_inverse": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = unique_consecutive_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('unique_consecutive', generated_inputs)
