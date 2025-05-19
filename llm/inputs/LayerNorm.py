
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def layernorm_inputs():
    list_of_inputs = []

    # Input 1: 2D tensor, normalized_shape = [2]
    input1 = np.random.randn(3, 2).astype(np.float32)
    input_dict1 = {
        "input": input1,
        "normalized_shape": [2],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor, normalized_shape = [3, 4]
    input2 = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict2 = {
        "input": input2,
        "normalized_shape": [3, 4],
        "eps": 1e-8,
        "elementwise_affine": False,
        "bias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 4D tensor with negative values, normalized_shape = [4]
    input3 = np.random.randn(1, 2, 3, 4).astype(np.float32) * -1
    input_dict3 = {
        "input": input3,
        "normalized_shape": [4],
        "eps": 1e-6,
        "elementwise_affine": True,
        "bias": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor, normalized_shape = [5]
    input4 = np.random.randn(5).astype(np.float32)
    input_dict4 = {
        "input": input4,
        "normalized_shape": [5],
        "eps": 1e-5,
        "elementwise_affine": False,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    # Input 5: 5D tensor, normalized_shape = [3, 4, 5]
    input5 = np.random.randn(1, 2, 3, 4, 5).astype(np.float32)
    input_dict5 = {
        "input": input5,
        "normalized_shape": [3, 4, 5],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6:  Float16 Tensor
    input6 = np.random.randn(2, 4).astype(np.float16)
    input_dict6 = {
        "input": input6,
        "normalized_shape": [4],
        "eps": 1e-5,
        "elementwise_affine": True,
        "bias": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = layernorm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('LayerNorm', generated_inputs)
