
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def MSELoss_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors, same shape
    input1 = np.random.randn(3, 4).astype(np.float32)
    target1 = np.random.randn(3, 4).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "target": target1,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: One dimensional tensor
    input4 = np.random.randn(5).astype(np.float32)
    target4 = np.random.randn(5).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "target": target4,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 3: Negative values
    input6 = np.random.randn(2, 2).astype(np.float32) * -1
    target6 = np.random.randn(2, 2).astype(np.float32) * -1
    input_dict6 = {
        "input": input6,
        "target": target6,
        "size_average": None,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Case 4: Different type
    input2 = np.random.randn(3, 4).astype(np.float64)
    target2 = np.random.randn(3, 4).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "target": target2,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Case 5: Float tensors, same shape
    input7 = np.random.randn(2, 3).astype(np.float32)
    target7 = np.random.randn(2, 3).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "target": target7,
        "size_average": True,
        "reduce": True,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = MSELoss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('MSELoss', generated_inputs)
