
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def atanh_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor within the valid range (-1, 1)
    input1 = np.array([-0.5, 0, 0.5]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor with negative values, close to -1 and 1
    input2 = np.array([-0.99, -0.75, -0.25, 0.25, 0.75, 0.99]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D float tensor
    input3 = np.array([[-0.8, 0.2], [0.4, -0.6]]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D float tensor
    input4 = np.random.uniform(low=-0.9, high=0.9, size=(2, 2, 2)).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Float16 tensor
    input5 = np.array([-0.1, 0.3, -0.5, 0.7]).astype(np.float16)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Scalar tensor
    input6 = np.array(0.2).astype(np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7: Array containing near-zero values.
    input7 = np.array([-0.0001, 0, 0.0001]).astype(np.float32)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = atanh_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('atanh_', generated_inputs)
