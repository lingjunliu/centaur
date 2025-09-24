
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def inner_inputs():
    list_of_inputs = []

    # Test case 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0])
    other1 = np.array([4.0, 5.0, 6.0])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: Integer tensors
    input2 = np.array([1, 2, 3], dtype=np.int32)
    other2 = np.array([4, 5, 6], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: Multi-dimensional tensors
    input3 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other3 = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Test case 4: Negative values
    input4 = np.array([-1.0, -2.0, -3.0])
    other4 = np.array([4.0, 5.0, 6.0])
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: Different shapes (but compatible for inner product)
    input5 = np.array([[1.0, 2.0, 3.0]])
    other5 = np.array([4.0, 5.0, 6.0])
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 6: Complex tensors
    input6 = np.array([1+1j, 2+2j, 3+3j])
    other6 = np.array([4+4j, 5+5j, 6+6j])
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Test case 7: 3D tensors, compatible shapes
    input7 = np.random.rand(2, 3, 4)
    other7 = np.random.rand(2, 3, 4)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = inner_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('inner', generated_inputs)
