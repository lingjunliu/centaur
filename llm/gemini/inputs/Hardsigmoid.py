
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def hardsigmoid_inputs():
    list_of_inputs = []

    input1 = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.float32)
    input_dict1 = {
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[-4, -3, -2], [-1, 0, 1], [2, 3, 4]], dtype=np.float64)
    input_dict2 = {
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    input_dict3 = {
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.array([[-5.5, -3.2, 0.1], [2.8, 3.1, 6.2]], dtype=np.float16)
    input_dict4 = {
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([-10, 0, 10], dtype=np.float32)
    input_dict5 = {
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict6 = {
        "inplace": True,
        "input": input6.astype(np.float32) 
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs = hardsigmoid_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Hardsigmoid', generated_inputs)
