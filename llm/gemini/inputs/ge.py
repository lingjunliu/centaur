
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def ge_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other1 = np.array([[2.0, 1.0], [4.0, 3.0]])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensors with negative values
    input2 = np.array([[-1, 0], [1, 2]], dtype=np.int32)
    other2 = np.array([[0, -1], [2, 1]], dtype=np.int32)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different shapes (input is a scalar)
    input3 = np.array(5.0)
    other3 = np.array([[4.0, 6.0], [5.0, 3.0]])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different shapes (other is a scalar)
    input4 = np.array([[4.0, 6.0], [5.0, 3.0]])
    other4 = np.array(5.0)
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 3D tensors
    input5 = np.random.rand(2, 3, 4)
    other5 = np.random.rand(2, 3, 4)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Boolean tensors
    input6 = np.array([[True, False], [False, True]])
    other6 = np.array([[False, True], [True, False]])
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Zero dimension tensors (scalar)
    input7 = np.array(3)
    other7 = np.array(2)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = ge_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ge', generated_inputs)
