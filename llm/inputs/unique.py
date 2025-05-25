
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def unique_inputs():
    list_of_inputs = []

    # Input 1: 1D integer tensor
    input1 = np.array([1, 3, 2, 3, 1], dtype=np.int64)
    input_dict1 = {
        "input": input1,
        "sorted": True,
        "return_inverse": False,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with return_inverse and return_counts
    input2 = np.array([[1.5, 2.5], [3.5, 1.5], [2.5, 4.5]], dtype=np.float32)
    input_dict2 = {
        "input": input2,
        "sorted": False,
        "return_inverse": True,
        "return_counts": True,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D integer tensor with dim=0
    input3 = np.array([[[1, 2], [3, 4]], [[1, 2], [5, 6]], [[7, 8], [9, 10]]], dtype=np.int32)
    input_dict3 = {
        "input": input3,
        "sorted": True,
        "return_inverse": False,
        "return_counts": False,
        "dim": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D integer tensor with dim=1
    input4 = np.array([[[1, 2], [3, 4]], [[1, 2], [5, 6]], [[7, 8], [9, 10]]], dtype=np.int32)
    input_dict4 = {
        "input": input4,
        "sorted": False,
        "return_inverse": True,
        "return_counts": True,
        "dim": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D integer tensor with dim=2
    input5 = np.array([[[1, 2], [3, 4]], [[1, 2], [5, 6]], [[7, 8], [9, 10]]], dtype=np.int32)
    input_dict5 = {
        "input": input5,
        "sorted": True,
        "return_inverse": False,
        "return_counts": False,
        "dim": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 1D tensor with negative values
    input6 = np.array([-1, -2, -1, 0, 1, 2, 0], dtype=np.int64)
    input_dict6 = {
        "input": input6,
        "sorted": True,
        "return_inverse": True,
        "return_counts": True,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: 2D tensor with mixed positive and negative float values
    input7 = np.array([[-1.5, 2.5], [3.5, -1.5], [2.5, 4.5]], dtype=np.float64)
    input_dict7 = {
        "input": input7,
        "sorted": True,
        "return_inverse": False,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = unique_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('unique', generated_inputs)
