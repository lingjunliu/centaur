
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def adaptive_max_pool2d_inputs():
    list_of_inputs = []

    input1 = np.random.randn(2, 3, 20, 20).astype(np.float32)
    output_size1 = (5, 7)
    input_dict1 = {"input": input1, "output_size": output_size1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(1, 1, 32, 32).astype(np.float64)
    output_size2 = (10, 10)
    input_dict2 = {"input": input2, "output_size": output_size2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(4, 5, 16, 16).astype(np.float16)
    output_size3 = (8, 8)
    input_dict3 = {"input": input3, "output_size": output_size3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = np.random.randn(3, 7, 24, 24).astype(np.float32)
    output_size4 = (12, 6)
    input_dict4 = {"input": input4, "output_size": output_size4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.randn(1, 2, 28, 28).astype(np.float64)
    output_size5 = (14, 7)
    input_dict5 = {"input": input5, "output_size": output_size5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.randn(2, 4, 10, 10).astype(np.float32)
    output_size6 = 7
    input_dict6 = {"input": input6, "output_size": output_size6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.randn(1, 3, 15, 15).astype(np.float64)
    output_size7 = 5
    input_dict7 = {"input": input7, "output_size": output_size7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = adaptive_max_pool2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('adaptive_max_pool2d', generated_inputs)
