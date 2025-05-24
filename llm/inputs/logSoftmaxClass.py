
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def log_softmax_inputs():
    list_of_inputs = []

    input1 = np.random.randn(2, 3).astype(np.float32)
    input_dict1 = {"input": input1, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(5, 4, 6).astype(np.float64)
    input_dict2 = {"input": input2, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(1, 5, 3, 2).astype(np.float32)
    input_dict3 = {"input": input3, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(2, 2).astype(np.float32)
    input_dict4 = {"input": input4, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.randn(3, 5, 2).astype(np.float64)
    input_dict5 = {"input": input5, "dim": -2}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.randn(4).astype(np.float32)
    input_dict6 = {"input": input6, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict7 = {"input": input7, "dim": 3}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    input8 = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float32)
    input_dict8 = {"input": input8, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs = log_softmax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('logSoftmaxClass', generated_inputs)
