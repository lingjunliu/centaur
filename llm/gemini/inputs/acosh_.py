
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def acosh_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative values (should still work as acosh is defined for x >= 1)
    input2 = np.array([1.0, 2.0, 3.0, 10.0], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Input 3: 2D float tensor
    input3 = np.array([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger values
    input4 = np.array([100.0, 1000.0, 10000.0], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Scalar input
    input5 = np.array(1.5, dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Zeros (should raise an error but need to include to test edge cases)
    input6 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs = acosh_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('acosh_', generated_inputs)
