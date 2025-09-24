
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def std_mean_inputs():
    list_of_inputs = []

    input1 = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "dim": 1,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "dim": 0,
        "correction": 1,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input5 = np.random.randn(1, 5).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "dim": 1,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict6 = {
        "input": input6,
        "dim": (0, 2),
        "correction": 2,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "dim": None,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.random.randn(4).astype(np.float32)
    input_dict8 = {
        "input": input8,
        "dim": 0,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    input9 = np.array([[-1.0, -2.0], [-3.0, -4.0]]).astype(np.float32)
    input_dict9 = {
        "input": input9,
        "dim": 1,
        "correction": 0,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    input10 = np.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j, 4.0 + 4j]).astype(np.complex64)
    input_dict10 = {
        "input": input10,
        "dim": 0,
        "correction": 0,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
        
    return list_of_inputs

generated_inputs = std_mean_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('std_mean', generated_inputs)
