
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def dsplit_inputs():
    list_of_inputs = []

    # Case 1: Basic integer split
    input_tensor = torch.arange(24.0).reshape(2, 3, 4).numpy()
    indices_or_sections = 2
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer split with different dimensions
    input_tensor = torch.arange(36.0).reshape(3, 2, 6).numpy()
    indices_or_sections = 3
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer split with float tensor
    input_tensor = torch.randn(2, 2, 8).numpy()
    indices_or_sections = 4
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: List of indices
    input_tensor = torch.arange(36.0).reshape(3, 4, 3).numpy()
    indices_or_sections = [1]
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Three dimensions, indices or sections = 1
    input_tensor = torch.arange(24.0).reshape(2, 3, 4).numpy()
    indices_or_sections = 1
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.dsplit_1"] = dsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.dsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.dsplit_1'.")

check_valid('torch.dsplit', generated_inputs['torch.dsplit_1'], lib="torch")
