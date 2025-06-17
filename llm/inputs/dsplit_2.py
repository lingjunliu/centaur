
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def dsplit_inputs():
    list_of_inputs = []

    # Case 1: Basic 3D tensor, split into 2
    t = torch.arange(16.0).reshape(2, 2, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Basic 3D tensor, uneven split
    t = torch.arange(16.0).reshape(2, 2, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1, 3]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer input
    t = torch.arange(24.0).reshape(2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 4D tensor
    t = torch.arange(48.0).reshape(2, 2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Uneven split on 4D Tensor
    t = torch.arange(48.0).reshape(2, 2, 3, 4).numpy()
    input_dict = {"input": t, "indices_or_sections": [1, 2]}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dsplit_2"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dsplit_2'.")

check_valid('torch.dsplit', generated_inputs['torch.dsplit_2'], lib="torch")
