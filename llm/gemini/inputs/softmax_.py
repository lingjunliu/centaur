
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def softmax_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor, dim=1
    input1 = np.random.randn(3, 5)
    input_dict1 = {"input": input1, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor, dim=0
    input2 = np.random.randn(2, 3, 4)
    input_dict2 = {"input": input2, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor, dim=0 (only option)
    input3 = np.random.randn(5)
    input_dict3 = {"input": input3, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with negative values, dim=-1
    input4 = np.random.randn(4, 6) * -1
    input_dict4 = {"input": input4, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D tensor, dim=2
    input5 = np.random.randn(2, 3, 4, 5)
    input_dict5 = {"input": input5, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different shape tensor, dim=-2
    input6 = np.random.randn(5, 2, 3)
    input_dict6 = {"input": input6, "dim": -2}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Large tensor, dim=0
    input7 = np.random.randn(10, 10, 10)
    input_dict7 = {"input": input7, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = softmax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('softmax_', generated_inputs)
