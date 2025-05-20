
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: 2D tensor, dim=1
    input1 = np.random.randn(2, 3).astype(np.float32)
    input_dict1 = {"dim": 1, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor, dim=0
    input2 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict2 = {"dim": 0, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor with negative values, dim=0
    input3 = np.random.randn(5, 5).astype(np.float32) * -1
    input_dict3 = {"dim": 0, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor, dim=0
    input4 = np.random.randn(10).astype(np.float32)
    input_dict4 = {"dim": 0, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D tensor, dim=2
    input5 = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict5 = {"dim": 2, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 3D tensor, dim=1
    input6 = np.random.randn(3, 5, 2).astype(np.float32)
    input_dict6 = {"dim": 1, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = log_softmax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('LogSoftmax', generated_inputs)
