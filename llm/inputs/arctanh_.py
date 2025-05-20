
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def arctanh_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([0.1, 0.5, -0.2, 0.8]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Multi-dimensional float tensor with values near 1 and -1 (but not equal)
    input2 = np.array([[0.9, -0.95], [0.7, -0.65]]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor with a wider range of values (still within -1 and 1)
    input3 = np.linspace(-0.99, 0.99, 10).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Higher dimensional float tensor
    input4 = np.random.uniform(-0.9, 0.9, size=(2, 3, 4)).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Single element float tensor
    input5 = np.array([0.5]).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: Another set of different float values
    input6 = np.array([-0.3, 0.6, -0.8, 0.2]).astype(np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: A 3D tensor with more diverse values
    input7 = np.random.uniform(-0.7, 0.7, size=(3, 2, 2)).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = arctanh_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('arctanh_', generated_inputs)
