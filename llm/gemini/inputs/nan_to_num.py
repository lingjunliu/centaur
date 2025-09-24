
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def nan_to_num_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor with NaN, inf, -inf
    input1 = np.array([float('nan'), float('inf'), float('-inf'), 1.0, 2.0], dtype=np.float32)
    input_dict1 = {
        "input": input1,
        "nan": 0.0,
        "posinf": 1.0,
        "neginf": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Multi-dimensional float tensor
    input3 = np.array([[float('nan'), 1.0], [2.0, float('inf')]], dtype=np.float64)
    input_dict3 = {
        "input": input3,
        "nan": -1.0,
        "posinf": 10.0,
        "neginf": -10.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 3: Tensor with different replacement values
    input4 = np.array([float('nan'), float('inf'), float('-inf'), 0.0], dtype=np.float32)
    input_dict4 = {
        "input": input4,
        "nan": 100.0,
        "posinf": 200.0,
        "neginf": 300.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 4: Tensor with zero replacement values
    input5 = np.array([float('nan'), float('inf'), float('-inf')], dtype=np.float32)
    input_dict5 = {
        "input": input5,
        "nan": 0.0,
        "posinf": 0.0,
        "neginf": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 5: 3D tensor
    input6 = np.array([[[float('nan'), 1.0], [2.0, float('inf')]], [[3.0, float('-inf')], [4.0, 5.0]]], dtype=np.float32)
    input_dict6 = {
        "input": input6,
        "nan": 0.5,
        "posinf": 10.5,
        "neginf": -10.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = nan_to_num_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('nan_to_num', generated_inputs)
