
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def hsplit_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor, indices_or_sections is an integer
    input_tensor = torch.arange(16.0).reshape(4, 4).numpy()
    indices_or_sections = 2
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int tensor, indices_or_sections is an integer
    input_tensor = torch.arange(12).reshape(3, 4).numpy()
    indices_or_sections = 4
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float tensor, indices_or_sections is a list
    input_tensor = torch.randn(2, 5).numpy()
    indices_or_sections = [2, 4]
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int tensor, indices_or_sections is an integer
    input_tensor = torch.randint(0, 10, (2, 4, 4)).numpy()
    indices_or_sections = 2
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float tensor, indices_or_sections is a list
    input_tensor = torch.randn(3, 6, 4).numpy()
    indices_or_sections = [2, 5]
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int tensor, indices_or_sections is an integer
    input_tensor = torch.arange(6).numpy()
    indices_or_sections = 3
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float tensor, indices_or_sections is a list
    input_tensor = torch.randn(8).numpy()
    indices_or_sections = [2, 5]
    input_dict = {"input": input_tensor, "indices_or_sections": indices_or_sections}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.hsplit_1"] = hsplit_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.hsplit_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.hsplit_1'.")

check_valid('torch.hsplit', generated_inputs['torch.hsplit_1'], lib="torch")
