
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def arcsinh_inputs():
    generated_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    generated_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative values
    input2 = np.array([-1.0, -2.0, 0.5, -0.5], dtype=np.float64)
    input_dict2 = {"input": input2}
    generated_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional tensor
    input3 = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    input_dict3 = {"input": input3}
    generated_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger values
    input4 = np.array([10.0, 20.0, -10.0, -20.0], dtype=np.float64)
    input_dict4 = {"input": input4}
    generated_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Zero tensor
    input5 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    input_dict5 = {"input": input5}
    generated_inputs.append(copy.deepcopy(input_dict5))

    # Input 6:  Different dimension
    input6 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    input_dict6 = {"input": input6}
    generated_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Smaller values
    input7 = np.array([0.1, 0.2, -0.1, -0.2], dtype=np.float64)
    input_dict7 = {"input": input7}
    generated_inputs.append(copy.deepcopy(input_dict7))

    return generated_inputs

generated_inputs = arcsinh_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('arcsinh', generated_inputs)
