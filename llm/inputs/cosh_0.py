
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def cosh_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, 1D
    input1 = torch.randn(5).numpy()
    input_dict1 = {"input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Float tensor, 2D, with negative values
    input2 = torch.randn(3, 4).numpy()
    input_dict2 = {"input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor, 3D
    input3 = torch.randn(2, 3, 2).numpy()
    input_dict3 = {"input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs["torch.cosh"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cosh'.")

check_valid('torch.cosh', generated_inputs['torch.cosh'], lib="torch")
