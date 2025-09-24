
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

generated_inputs = dict()

import torch, copy
import numpy as np

def layer_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float input, 2D
    input1 = np.random.randn(2, 3).astype(np.float32)
    normalized_shape1 = (3,)
    weight1 = np.random.randn(3).astype(np.float32)
    bias1 = np.random.randn(3).astype(np.float32)
    eps1 = 1e-5
    input_dict1 = {
        "input": input1,
        "normalized_shape": normalized_shape1,
        "weight": weight1,
        "bias": bias1,
        "eps": eps1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D input, different normalized shape
    input2 = np.random.randn(2, 4, 5).astype(np.float32)
    normalized_shape2 = (4, 5)
    weight2 = np.random.randn(4, 5).astype(np.float32)
    bias2 = np.random.randn(4, 5).astype(np.float32)
    eps2 = 1e-8
    input_dict2 = {
        "input": input2,
        "normalized_shape": normalized_shape2,
        "weight": weight2,
        "bias": bias2,
        "eps": eps2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Input with negative values, 1D
    input3 = np.random.randn(10).astype(np.float32) - 5
    normalized_shape3 = (10,)
    weight3 = np.random.randn(10).astype(np.float32)
    bias3 = np.random.randn(10).astype(np.float32)
    eps3 = 1e-6
    input_dict3 = {
        "input": input3,
        "normalized_shape": normalized_shape3,
        "weight": weight3,
        "bias": bias3,
        "eps": eps3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger input, 4D
    input4 = np.random.randn(1, 3, 16, 16).astype(np.float32)
    normalized_shape4 = (16, 16)
    weight4 = np.random.randn(16, 16).astype(np.float32)
    bias4 = np.random.randn(16, 16).astype(np.float32)
    eps4 = 1e-7
    input_dict4 = {
        "input": input4,
        "normalized_shape": normalized_shape4,
        "weight": weight4,
        "bias": bias4,
        "eps": eps4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different eps
    input5 = np.random.randn(2, 3).astype(np.float32)
    normalized_shape5 = (3,)
    weight5 = np.random.randn(3).astype(np.float32)
    bias5 = np.random.randn(3).astype(np.float32)
    eps5 = 1e-3
    input_dict5 = {
        "input": input5,
        "normalized_shape": normalized_shape5,
        "weight": weight5,
        "bias": bias5,
        "eps": eps5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 1D input
    input6 = np.random.randn(5).astype(np.float32)
    normalized_shape6 = (5,)
    weight6 = np.random.randn(5).astype(np.float32)
    bias6 = np.random.randn(5).astype(np.float32)
    eps6 = 1e-4
    input_dict6 = {
        "input": input6,
        "normalized_shape": normalized_shape6,
        "weight": weight6,
        "bias": bias6,
        "eps": eps6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.layer_norm"] = layer_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('layer_norm', generated_inputs['torch.nn.functional.layer_norm'], lib="torch")
