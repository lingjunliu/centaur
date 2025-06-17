
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def asin_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = np.array([-0.5, 0, 0.5]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Negative and positive values within [-1, 1]
    input2 = np.array([-1, -0.75, 0, 0.25, 0.75, 1]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Multi-dimensional tensor
    input3 = np.array([[-0.8, 0.2], [0.5, -0.1]]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    return list_of_inputs

generated_inputs["torch.asin"] = asin_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.asin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.asin'.")

check_valid('torch.asin', generated_inputs['torch.asin'], lib="torch")
