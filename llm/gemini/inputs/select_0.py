
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_select_inputs():
    list_of_inputs = []

    # Input 1: 2D float tensor
    input_tensor = torch.randn(3, 4).numpy()
    dim = 0
    index = 1
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D int tensor
    input_tensor = torch.randint(0, 10, (2, 3, 5)).numpy()
    dim = 2
    index = 3
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D tensor with negative index
    input_tensor = torch.arange(5).float().numpy()
    dim = 0
    index = -1
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D tensor
    input_tensor = torch.randn(2, 3, 4, 5).numpy()
    dim = 1
    index = 0
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D tensor with negative dim
    input_tensor = torch.randn(5, 5).numpy()
    dim = -1
    index = 2
    input_dict = {"input": input_tensor, "dim": dim, "index": index}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.select"] = torch_select_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.select' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.select'.")

check_valid('torch.select', generated_inputs['torch.select'], lib="torch")
