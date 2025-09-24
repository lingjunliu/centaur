
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def flatten_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, default start_dim and end_dim
    input1 = np.random.randn(2, 3).astype(np.float32)
    input_dict1 = {"input": input1, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D int tensor, specified start_dim and end_dim
    input2 = np.random.randint(0, 10, size=(1, 4, 5)).astype(np.int32)
    input_dict2 = {"input": input2, "start_dim": 1, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D float tensor, specified start_dim and end_dim
    input3 = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict3 = {"input": input3, "start_dim": 1, "end_dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D float tensor
    input4 = np.random.randn(10).astype(np.float32)
    input_dict4 = {"input": input4, "start_dim": 0, "end_dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 5D int tensor, flatten all dimensions
    input5 = np.random.randint(0, 5, size=(1, 2, 3, 4, 5)).astype(np.int64)
    input_dict5 = {"input": input5, "start_dim": 0, "end_dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = flatten_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Flatten', generated_inputs)
