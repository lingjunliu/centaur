
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_diagonal_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D tensor
    input1 = torch.randn(3, 3).numpy()
    input_dict1 = {"input": input1, "offset": 0, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: 2D tensor with offset
    input2 = torch.randn(4, 4).numpy()
    input_dict2 = {"input": input2, "offset": 1, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 2D tensor with negative offset
    input3 = torch.randn(5, 5).numpy()
    input_dict3 = {"input": input3, "offset": -1, "dim1": 0, "dim2": 1}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 3D tensor with different dims
    input4 = torch.randn(2, 5, 4).numpy()
    input_dict4 = {"input": input4, "offset": 0, "dim1": 1, "dim2": 2}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: 4D tensor
    input5 = torch.randn(2, 3, 4, 5).numpy()
    input_dict5 = {"input": input5, "offset": 1, "dim1": 2, "dim2": 3}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.diagonal"] = torch_diagonal_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.diagonal' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.diagonal'.")

check_valid('torch.diagonal', generated_inputs['torch.diagonal'], lib="torch")
