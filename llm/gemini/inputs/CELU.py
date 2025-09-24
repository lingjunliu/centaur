
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def celu_inputs():
    list_of_inputs = []

    input1 = np.random.randn(2).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(2, 3).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "alpha": 0.5,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.randn(1, 2, 3).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "alpha": 2.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.randn(1, 2, 3, 4).astype(np.float64)
    input_dict4 = {
        "input": input4,
        "alpha": 0.75,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = np.random.randn(5).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "alpha": 1.25,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "alpha": 1.5,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.random.randn(2, 2, 2, 2, 2).astype(np.float64)
    input_dict7 = {
        "input": input7,
        "alpha": 0.25,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = celu_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('CELU', generated_inputs)
