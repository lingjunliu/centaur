
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def greater_equal_inputs():
    list_of_inputs = []

    # Test case 1: Float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other1 = np.array([[2.0, 1.0], [3.0, 5.0]])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Integer tensors
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    other2 = np.array([[2, 1], [3, 5]], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Scalar comparison
    input3 = np.array([[1, 2], [3, 4]])
    other3 = 3
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Multi-dimensional tensors
    input4 = np.random.rand(2, 3, 4)
    other4 = np.random.rand(2, 3, 4)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Negative values
    input5 = np.array([[-1, -2], [-3, -4]])
    other5 = np.array([[0, -1], [-3, -5]])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: Different shapes (broadcasting)
    input6 = np.array([[1, 2, 3]])
    other6 = np.array([2, 1, 4])
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: Boolean tensor
    input7 = np.array([[True, False], [True, True]])
    other7 = np.array([[False, True], [True, False]])
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    # Test case 8: Zero dimension tensor
    input8 = np.array(5)
    other8 = np.array(3)
    input_dict8 = {"input": input8, "other": other8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = greater_equal_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('greater_equal', generated_inputs)
