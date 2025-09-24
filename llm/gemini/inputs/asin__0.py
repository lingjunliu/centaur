
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def asin__inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor within [-1, 1]
    input1 = np.array([-0.5, 0, 0.5]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor with boundary values
    input2 = np.array([-1, 1]).astype(np.float32)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional float tensor
    input3 = np.array([[-0.2, 0.3], [0.6, -0.9]]).astype(np.float64)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Larger multi-dimensional float tensor
    input4 = np.random.uniform(low=-1.0, high=1.0, size=(2, 3, 4)).astype(np.float32)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Scalar float tensor
    input5 = np.array(0.7).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.asin_"] = asin__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.asin_' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asin_'.")

check_valid('torch.asin_', generated_inputs['torch.asin_'], lib="torch")
