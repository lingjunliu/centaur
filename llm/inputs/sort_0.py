
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_sort_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor
    input_tensor = torch.randn(10).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "descending": False, "stable": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, sort along dimension 0, descending order
    input_tensor = torch.randn(5, 5).numpy()
    input_dict = {"input": input_tensor, "dim": 0, "descending": True, "stable": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, sort along dimension 1, stable sort
    input_tensor = torch.randint(0, 5, (3, 4, 2)).float().numpy()
    input_dict = {"input": input_tensor, "dim": 1, "descending": False, "stable": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D integer tensor, sort along dimension 1, descending and stable
    input_tensor = torch.randint(-5, 5, (4, 6)).numpy()
    input_dict = {"input": input_tensor, "dim": 1, "descending": True, "stable": True, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D tensor with negative values
    input_tensor = torch.randn(7).numpy() - 2
    input_dict = {"input": input_tensor, "dim": 0, "descending": False, "stable": False, "out": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["torch.sort"] = torch_sort_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.sort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.sort'.")

check_valid('torch.sort', generated_inputs['torch.sort'], lib="torch")
