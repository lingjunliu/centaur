
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def equal_inputs():
    list_of_inputs = []

    # Test case 1: Equal 1D integer tensors
    input1 = np.array([1, 2, 3])
    input2 = np.array([1, 2, 3])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Unequal 1D integer tensors
    input1 = np.array([1, 2, 3])
    input2 = np.array([1, 2, 4])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Equal 2D float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Unequal 2D float tensors with different shapes
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = np.array([[1.0, 2.0]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Equal 3D boolean tensors
    input1 = np.array([[[True, False], [True, True]], [[False, False], [True, False]]])
    input2 = np.array([[[True, False], [True, True]], [[False, False], [True, False]]])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 6: Unequal 1D tensors with different data types
    input1 = np.array([1, 2, 3], dtype=np.int32)
    input2 = np.array([1, 2, 3], dtype=np.float64)
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 7: Equal 1D complex tensors
    input1 = np.array([1+1j, 2+2j, 3+3j])
    input2 = np.array([1+1j, 2+2j, 3+3j])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 8: Unequal 1D complex tensors
    input1 = np.array([1+1j, 2+2j, 3+3j])
    input2 = np.array([1+1j, 2+2j, 4+4j])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 9: Equal 1D integer tensors with negative values
    input1 = np.array([-1, -2, 3])
    input2 = np.array([-1, -2, 3])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 10: Unequal 1D integer tensors with different values
    input1 = np.array([-1, -2, 3])
    input2 = np.array([-1, 2, 3])
    input_dict = {"input": input1, "other": input2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = equal_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('equal', generated_inputs)
