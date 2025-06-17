
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def layer_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float input with weight and bias
    input1 = np.random.randn(2, 3, 4).astype(np.float32)
    normalized_shape1 = (3, 4)
    weight1 = np.random.randn(4).astype(np.float32)
    bias1 = np.random.randn(4).astype(np.float32)
    eps1 = 1e-5
    input_dict1 = {
        "input": input1,
        "normalized_shape": normalized_shape1,
        "weight": weight1,
        "bias": bias1,
        "eps": eps1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer input without weight and bias
    input2 = np.random.randint(-5, 5, size=(1, 5, 5)).astype(np.float32) #changed to float32 to avoid error
    normalized_shape2 = (5, )
    weight2 = None
    bias2 = None
    eps2 = 1e-8
    input_dict2 = {
        "input": input2,
        "normalized_shape": normalized_shape2,
        "weight": weight2,
        "bias": bias2,
        "eps": eps2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Input with negative values and different normalized shape
    input3 = np.random.randn(4, 2, 2).astype(np.float32)
    normalized_shape3 = (2, )
    weight3 = np.random.randn(2).astype(np.float32)
    bias3 = np.random.randn(2).astype(np.float32)
    eps3 = 1e-6
    input_dict3 = {
        "input": input3,
        "normalized_shape": normalized_shape3,
        "weight": weight3,
        "bias": bias3,
        "eps": eps3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.layer_norm"] = layer_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.layer_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.layer_norm'.")

check_valid('torch.nn.functional.layer_norm', generated_inputs['torch.nn.functional.layer_norm'], lib="torch")
