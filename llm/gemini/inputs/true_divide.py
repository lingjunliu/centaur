
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def true_divide_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    input1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other1 = np.array([[0.5, 1.0], [1.5, 2.0]])
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensors
    input2 = np.array([[1, 2], [3, 4]])
    other2 = np.array([[2, 2], [2, 2]])
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values
    input3 = np.array([[-1.0, 2.0], [-3.0, 4.0]])
    other3 = np.array([[0.5, -1.0], [1.5, -2.0]])
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different shapes (broadcasting)
    input4 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    other4 = np.array([1.0, 2.0, 3.0])
    input_dict4 = {"input": input4, "other": other4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Scalar divisor
    input5 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other5 = np.array(2.0)
    input_dict5 = {"input": input5, "other": other5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Multidimensional arrays
    input6 = np.random.rand(2, 3, 4)
    other6 = np.random.rand(2, 3, 4)
    input_dict6 = {"input": input6, "other": other6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Zero values in divisor
    input7 = np.array([[1.0, 2.0], [3.0, 4.0]])
    other7 = np.array([[0.5, 0.0], [1.5, 0.0]])
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = true_divide_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('true_divide', generated_inputs)
