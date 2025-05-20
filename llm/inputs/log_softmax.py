
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def log_softmax_inputs():
    list_of_inputs = []

    # Input 1: 1D array, dim=0
    input1 = np.array([1.0, 2.0, 3.0])
    input_dict1 = {"input": input1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D array, dim=1
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    input_dict2 = {"input": input2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D array with negative values, dim=0
    input3 = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    input_dict3 = {"input": input3, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 3D array, dim=2
    input4 = np.random.rand(2, 3, 4)
    input_dict4 = {"input": input4, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D array, dim=3
    input5 = np.random.rand(2, 3, 4, 5)
    input_dict5 = {"input": input5, "dim": 3}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 2D integer array, dim=1
    input6 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    input_dict6 = {"input": input6, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 1D array with a different dim
    input7 = np.array([1.0, 2.0, 3.0])
    input_dict7 = {"input": input7, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = log_softmax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('log_softmax', generated_inputs)
