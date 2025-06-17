
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def layer_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float input and affine parameters
    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    normalized_shape1 = (4,)
    weight1 = np.random.randn(4).astype(np.float32)
    bias1 = np.random.randn(4).astype(np.float32)
    eps1 = 1e-5
    elementwise_affine1 = True
    input_dict1 = {
        "input": input1,
        "normalized_shape": normalized_shape1,
        "weight": weight1,
        "bias": bias1,
        "eps": eps1,
        "elementwise_affine": elementwise_affine1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Input with int data type
    input2 = np.random.randint(0, 10, size=(2, 3, 4)).astype(np.float32)
    normalized_shape2 = (4,)
    weight2 = np.random.randn(4).astype(np.float32)
    bias2 = np.random.randn(4).astype(np.float32)
    eps2 = 1e-5
    elementwise_affine2 = True
    input_dict2 = {
        "input": input2,
        "normalized_shape": normalized_shape2,
        "weight": weight2,
        "bias": bias2,
        "eps": eps2,
        "elementwise_affine": elementwise_affine2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Input with negative values and no affine parameters
    input3 = np.random.randn(2, 3, 4) * -1
    normalized_shape3 = (4,)
    weight3 = None
    bias3 = None
    eps3 = 1e-5
    elementwise_affine3 = False
    input_dict3 = {
        "input": input3,
        "normalized_shape": normalized_shape3,
        "weight": weight3,
        "bias": bias3,
        "eps": eps3,
        "elementwise_affine": elementwise_affine3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Input with different normalized_shape
    input4 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    normalized_shape4 = (4, 5)
    weight4 = np.random.randn(5).astype(np.float32)
    bias4 = np.random.randn(5).astype(np.float32)
    eps4 = 1e-5
    elementwise_affine4 = True
    input_dict4 = {
        "input": input4,
        "normalized_shape": normalized_shape4,
        "weight": weight4,
        "bias": bias4,
        "eps": eps4,
        "elementwise_affine": elementwise_affine4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Input with different eps value
    input5 = np.random.randn(2, 3, 4).astype(np.float32)
    normalized_shape5 = (4,)
    weight5 = np.random.randn(4).astype(np.float32)
    bias5 = np.random.randn(4).astype(np.float32)
    eps5 = 1e-8
    elementwise_affine5 = True
    input_dict5 = {
        "input": input5,
        "normalized_shape": normalized_shape5,
        "weight": weight5,
        "bias": bias5,
        "eps": eps5,
        "elementwise_affine": elementwise_affine5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    

    return list_of_inputs

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.layer_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.layer_norm'.")

check_valid('torch.nn.functional.layer_norm', generated_inputs['torch.nn.functional.layer_norm'], lib="torch")
