
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def atanh_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor with values in (-1, 1)
    input1 = np.array([-0.5, 0.2, 0.7, -0.9]).astype(np.float32)
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with values in (-1, 1)
    input2 = np.array([[0.1, -0.3], [0.6, -0.8]]).astype(np.float64)
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with values close to -1, 1, and within (-1, 1)
    input3 = np.array([[[0.5, -0.5], [0.2, -0.2]], [[0.3, -0.7], [0.1, -0.9]]]).astype(np.float32)
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: Single element tensor
    input4 = np.array(0.5).astype(np.float64)
    input_dict4 = {"input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor with mixed positive and negative values
    input5 = np.array([-0.2, 0.4, -0.6, 0.8]).astype(np.float32)
    input_dict5 = {"input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.atanh"] = atanh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.atanh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.atanh'.")

check_valid('torch.atanh', generated_inputs['torch.atanh'], lib="torch")
