
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def frexp_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    x = torch.arange(9.).float()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float tensor with negative values
    x = torch.randn(3, 4).float() * -1
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D float tensor
    x = torch.rand(2, 3, 5).float()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor with a large value
    x = torch.tensor([2**7], dtype=torch.float32).float()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D float tensor (scalar)
    x = torch.tensor(3.14).float()
    input_dict = {"input": x}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.frexp"] = frexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.frexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.frexp'.")

check_valid('torch.frexp', generated_inputs['torch.frexp'], lib="torch")
