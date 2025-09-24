
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_vsplit_inputs():
    list_of_inputs = []

    # Example 1: Splitting into equal parts
    input = torch.arange(16.0).reshape(4, 4).numpy()
    indices_or_sections = 2
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Splitting at specific indices
    input = torch.arange(20.0).reshape(5, 4).numpy()
    indices_or_sections = [2, 4]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 3D tensor
    input = torch.arange(24.0).reshape(2, 3, 4).numpy()
    indices_or_sections = [1]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4: Uneven split
    input = torch.arange(15.0).reshape(5, 3).numpy()
    indices_or_sections = [2, 4]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Float tensor with integer sections
    input = torch.randn(6, 2).numpy()
    indices_or_sections = 3
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 6: Different shaped input, single index
    input = torch.arange(12.0).reshape(6, 2).numpy()
    indices_or_sections = [4]
    input_dict = {"input": input, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.vsplit_2"] = torch_vsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.vsplit_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.vsplit_2'.")

check_valid('torch.vsplit', generated_inputs['torch.vsplit_2'], lib="torch")
