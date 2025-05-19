
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def layer_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, normalized shape is the last dimension
    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    normalized_shape1 = [4]
    input_dict1 = {
        "input": input1,
        "normalized_shape": normalized_shape1,
        "eps": 1e-5,
        "elementwise_affine": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor, normalized shape is multiple dimensions
    input2 = np.random.randn(2, 5, 5, 3).astype(np.float32)
    normalized_shape2 = [5, 3]
    input_dict2 = {
        "input": input2,
        "normalized_shape": normalized_shape2,
        "eps": 1e-8,
        "elementwise_affine": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values, different epsilon
    input3 = np.random.randn(1, 5, 7, 7).astype(np.float64) * -1
    normalized_shape3 = [7, 7]
    input_dict3 = {
        "input": input3,
        "normalized_shape": normalized_shape3,
        "eps": 1e-3,
        "elementwise_affine": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D input
    input4 = np.random.randn(10).astype(np.float32)
    normalized_shape4 = [10]
    input_dict4 = {
        "input": input4,
        "normalized_shape": normalized_shape4,
        "eps": 1e-5,
        "elementwise_affine": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger input tensor, different normalized shape
    input5 = np.random.randn(4, 6, 8, 10).astype(np.float32)
    normalized_shape5 = [10]
    input_dict5 = {
        "input": input5,
        "normalized_shape": normalized_shape5,
        "eps": 1e-6,
        "elementwise_affine": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = layer_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('layer_norm', generated_inputs)
