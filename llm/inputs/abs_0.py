
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_abs_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor with negative and positive integers
    input_tensor = torch.tensor([-1, -2, 3, -4, 5]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 2D tensor with floating-point numbers
    input_tensor = torch.tensor([[-1.5, 2.5], [-3.5, 4.5]]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 3D tensor with integers
    input_tensor = torch.tensor([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: 0D tensor (scalar)
    input_tensor = torch.tensor(-5).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: 1D tensor with only positive integers
    input_tensor = torch.tensor([1, 2, 3, 4, 5]).numpy()
    input_dict = {"input": input_tensor}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.abs"] = torch_abs_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.abs' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.abs'.")

check_valid('torch.abs', generated_inputs['torch.abs'], lib="torch")
