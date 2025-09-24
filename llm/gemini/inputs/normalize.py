
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def normalize_inputs():
    list_of_inputs = []

    input1 = np.random.rand(3, 4).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "dim": 1,
        "eps": 1e-12,
        "p": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "dim": 0,
        "eps": 1e-8,
        "p": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = np.random.rand(5,).astype(np.float32)
    input_dict3 = {
        "input": input3,
        "dim": 0,
        "eps": 1e-6,
        "p": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = np.random.rand(2, 2, 2, 2).astype(np.float64)
    input_dict4 = {
        "input": input4,
        "dim": 2,
        "eps": 1e-10,
        "p": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = (np.random.rand(3, 4) - 0.5).astype(np.float32) #negative values
    input_dict5 = {
        "input": input5,
        "dim": 1,
        "eps": 1e-12,
        "p": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = np.random.rand(1, 5).astype(np.float64)
    input_dict6 = {
        "input": input6,
        "dim": 1,
        "eps": 1e-8,
        "p": 1.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = np.array([[1,2],[3,4]]).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "dim": 0,
        "eps": 1e-6,
        "p": float('inf')
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))
    
    return list_of_inputs

generated_inputs = normalize_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('normalize', generated_inputs)
