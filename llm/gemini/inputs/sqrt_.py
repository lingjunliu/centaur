
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def sqrt__inputs():
    list_of_inputs = []

    input1 = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([[[1.0, 4.0], [9.0, 16.0]], [[25.0, 36.0], [49.0, 64.0]]], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([1, 4, 9, 16], dtype=np.int32)
    input_dict4 = {"input": input4.astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = np.array([[1, 4], [9, 16]], dtype=np.int64)
    input_dict5 = {"input": input5.astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([1.0], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = sqrt__inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('sqrt_', generated_inputs)
