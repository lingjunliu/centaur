
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_transpose_inputs():
    list_of_inputs = []

    # Example 1: Basic 2D tensor
    x = torch.randn(2, 3).numpy()
    input_dict = {"input": x, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 3D tensor
    x = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": x, "dim0": 1, "dim1": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Different dimensions
    x = torch.randn(5, 2).numpy()
    input_dict = {"input": x, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Integer tensor
    x = torch.randint(0, 10, (3, 5)).numpy()
    input_dict = {"input": x, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Float64 tensor
    x = torch.randn(2, 3, dtype=torch.float64).numpy()
    input_dict = {"input": x, "dim0": 0, "dim1": 1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: 4D tensor
    x = torch.randn(2, 3, 4, 5).numpy()
    input_dict = {"input": x, "dim0": 1, "dim1": 3}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 7: Negative indices
    x = torch.randn(2, 3).numpy()
    input_dict = {"input": x, "dim0": -2, "dim1": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.transpose"] = torch_transpose_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.transpose'.")

check_valid('torch.transpose', generated_inputs['torch.transpose'], lib="torch")
