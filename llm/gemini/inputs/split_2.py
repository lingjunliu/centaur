
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_split_inputs():
    list_of_inputs = []

    # Case 1: Integer split_size, default dim=0, 2D tensor
    tensor = np.arange(10).reshape(5, 2)
    split_size_or_sections = 2
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: List of split sizes, default dim=0, 2D tensor
    tensor = np.arange(10).reshape(5, 2)
    split_size_or_sections = [1, 4]
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer split_size, dim=1, 2D tensor
    tensor = np.arange(10).reshape(2, 5)
    split_size_or_sections = 2
    dim = 1
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: List of split sizes, dim=1, 2D tensor, unequal split
    tensor = np.arange(10).reshape(2, 5)
    split_size_or_sections = [2, 3]
    dim = 1
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Case 5: Integer split_size, dim=0, 3D tensor
    tensor = np.arange(24).reshape(2, 3, 4)
    split_size_or_sections = 1
    dim = 0
    input_dict = {"tensor": tensor, "split_size_or_sections": split_size_or_sections, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.split_2"] = torch_split_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.split_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.split_2'.")

check_valid('torch.split', generated_inputs['torch.split_2'], lib="torch")
