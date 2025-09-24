
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def column_stack_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data types (int and float)
    a = np.array([1, 2, 3])
    b = np.array([4.0, 5.0, 6.0])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensors
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Mix of 1D and 2D tensors (valid if broadcastable)
    a = np.array([1, 2])
    b = np.array([[3, 4], [5, 6]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensors with compatible shapes for column stacking
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensors with negative values
    a = np.array([-1, -2, -3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: More complex 2D arrays
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8, 9], [10, 11, 12]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = column_stack_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('column_stack', generated_inputs)
