
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def PReLU_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "num_parameters": 1,
        "init": 0.25
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict2 = {
        "input": input2,
        "num_parameters": 1,
        "init": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(1, 5, 5, 5).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "num_parameters": 1,
        "init": -0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "num_parameters": 1,
        "init": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.randn(4).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "num_parameters": 1,
        "init": 0.01
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "num_parameters": 3,
        "init": 0.25
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "num_parameters": 3,
        "init": -0.1
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = PReLU_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('PReLU_', generated_inputs)
