
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def where_inputs():
    list_of_inputs = []

    # Case 1: Basic case with boolean condition and float tensors
    condition = np.array([[True, False], [False, True]])
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    other_tensor = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors and condition
    condition = np.array([[True, False, True], [False, True, False]])
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]])
    other_tensor = np.array([[7, 8, 9], [10, 11, 12]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Different shapes (condition matches input/other)
    condition = np.array([True, False, True])
    input_tensor = np.array([1.0, 2.0, 3.0])
    other_tensor = np.array([4.0, 5.0, 6.0])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 3D tensors
    condition = np.array([[[True, False], [False, True]], [[False, True], [True, False]]])
    input_tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    other_tensor = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Negative values
    condition = np.array([[True, False], [False, True]])
    input_tensor = np.array([[-1.0, 2.0], [3.0, -4.0]])
    other_tensor = np.array([[5.0, -6.0], [-7.0, 8.0]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 6: condition as a numpy array of integers (0 and 1)
    condition = np.array([[1, 0], [0, 1]])
    input_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    other_tensor = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 7: Single element tensors
    condition = np.array(True)
    input_tensor = np.array(10.0)
    other_tensor = np.array(20.0)
    input_dict = {"condition": condition, "input": input_tensor, "other": other_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = where_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('where', generated_inputs)
