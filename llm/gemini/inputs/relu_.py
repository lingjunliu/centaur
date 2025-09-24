
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def relu_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor with negative values
    input1 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D integer tensor
    input2 = np.array([-1, 0, 1, 2, -3], dtype=np.int32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor
    input3 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: tensor with all zeros
    input4 = np.zeros((5, 5), dtype=np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: tensor with mixed positive and negative values, large and small
    input5 = np.array([[-100.0, 0.001], [1.0, -0.0001], [10.0, -1.0]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Higher dimensional tensor (4D)
    input6 = np.random.randn(1, 2, 3, 4).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Empty tensor
    input7 = np.array([], dtype=np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = relu_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('relu_', generated_inputs)
