
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def layer_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float32 and normalized_shape as a list
    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    normalized_shape1 = [4]
    eps1 = 1e-5
    elementwise_affine1 = True
    input_dict1 = {
        "input": input1,
        "normalized_shape": normalized_shape1,
        "eps": eps1,
        "elementwise_affine": elementwise_affine1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different normalized_shape and input dimensions, elementwise_affine=False
    input2 = np.random.randn(1, 5, 6, 7).astype(np.float32)
    normalized_shape2 = [6, 7]
    eps2 = 1e-8
    elementwise_affine2 = False
    input_dict2 = {
        "input": input2,
        "normalized_shape": normalized_shape2,
        "eps": eps2,
        "elementwise_affine": elementwise_affine2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))
    
    # Input 3:  Using a different data type (float64) and negative values
    input3 = (np.random.randn(3, 2, 5) - 0.5).astype(np.float64)  
    normalized_shape3 = [5]
    eps3 = 1e-6
    elementwise_affine3 = True
    input_dict3 = {
        "input": input3,
        "normalized_shape": normalized_shape3,
        "eps": eps3,
        "elementwise_affine": elementwise_affine3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D Input
    input4 = np.random.randn(10).astype(np.float32)
    normalized_shape4 = [10]
    eps4 = 1e-5
    elementwise_affine4 = True
    input_dict4 = {
        "input": input4,
        "normalized_shape": normalized_shape4,
        "eps": eps4,
        "elementwise_affine": elementwise_affine4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Smaller epsilon
    input5 = np.random.randn(2, 3, 4).astype(np.float32)
    normalized_shape5 = [4]
    eps5 = 1e-12
    elementwise_affine5 = True
    input_dict5 = {
        "input": input5,
        "normalized_shape": normalized_shape5,
        "eps": eps5,
        "elementwise_affine": elementwise_affine5
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
