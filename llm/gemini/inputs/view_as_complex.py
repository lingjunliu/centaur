
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def view_as_complex_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor
    input1 = np.random.randn(2, 2, 2).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D float tensor
    input2 = np.random.randn(2, 3, 2, 2).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Input 3: 1D float tensor
    input3 = np.random.randn(4,2).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different sized 2D tensor
    input4 = np.random.randn(5, 2, 2).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: Larger tensor with different data type
    input5 = np.random.randn(3, 4, 2, 2).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Another shape
    input6 = np.random.randn(2, 5, 2).astype(np.float64)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = view_as_complex_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('view_as_complex', generated_inputs)
