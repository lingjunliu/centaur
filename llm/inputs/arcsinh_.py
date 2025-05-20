
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def arcsinh_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, 1D
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor, 2D
    input2 = np.array([[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor, multi-dimensional with a zero
    input4 = np.array([[[0.0, 1.0], [-1.0, 0.0]], [[2.0, -2.0], [-3.0, 3.0]]], dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 4: Float tensor, larger values
    input5 = np.array([[-10.0, 10.0], [5.0, -5.0]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 5: Float tensor, scalar
    input6 = np.array(0.7, dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = arcsinh_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('arcsinh_', generated_inputs)
