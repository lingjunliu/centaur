
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cosh__inputs():
    list_of_inputs = []

    input1 = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.array([[0.0, 0.5], [1.0, -0.5]], dtype=np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0], dtype=np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.array([0.0 + 1.0j, 0.5 - 0.5j, -1.0 + 0.0j], dtype=np.complex64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)
    input_dict6 = {"input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float16)
    input_dict7 = {"input": input7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = cosh__inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cosh_', generated_inputs)
