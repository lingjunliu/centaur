
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def pdist_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D float tensor
    input1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    input_dict1 = {"input": input1, "p": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D float tensor with p=1.5
    input2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict2 = {"input": input2, "p": 1.5}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 5: 1D float tensor - Reshape to 2D
    input5 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2,2)
    input_dict5 = {"input": input5, "p": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6: 2D tensor with negative values
    input6 = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    input_dict6 = {"input": input6, "p": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Larger 2D float tensor
    input7 = np.random.rand(10, 5).astype(np.float32)
    input_dict7 = {"input": input7, "p": 2.0}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.pdist"] = pdist_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.pdist' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.pdist'.")

check_valid('torch.pdist', generated_inputs['torch.pdist'], lib="torch")
