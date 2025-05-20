
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def vstack_inputs():
    list_of_inputs = []

    # Test case 1: Two 1D tensors
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Two 2D tensors
    a = np.array([[1], [2], [3]])
    b = np.array([[4], [5], [6]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Multiple tensors with the same shape (after atleast_2d)
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    input_dict = {"tensors": [a, b, c]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Three 2D tensors with the same number of columns
    a = np.array([[1, 2, 3]])
    b = np.array([[4, 5, 6]])
    c = np.array([[7, 8, 9]])
    input_dict = {"tensors": [a, b, c]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Tensors with different data types (mixed int and float) - make sure they have the same shape
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6:  Tensors with negative values
    a = np.array([-1, -2, -3])
    b = np.array([4, 5, 6])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Three dimensional tensors with matching shapes
    a = np.array([[[1, 2], [3, 4]]])
    b = np.array([[[5, 6], [7, 8]]])
    input_dict = {"tensors": [a, b]}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = vstack_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('vstack', generated_inputs)
