
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def logcumsumexp_inputs():
    list_of_inputs = []

    # Test case 1: 1D tensor, dim=0
    a = torch.randn(10).numpy()
    dim = 0
    input_dict = {"input": a, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 2D tensor, dim=0
    a = torch.randn(5, 3).numpy()
    dim = 0
    input_dict = {"input": a, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: 2D tensor, dim=1
    a = torch.randn(5, 3).numpy()
    dim = 1
    input_dict = {"input": a, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: 3D tensor, dim=2
    a = torch.randn(2, 4, 5).numpy()
    dim = 2
    input_dict = {"input": a, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: 1D tensor with negative values, dim=0
    a = (torch.randn(10) * -1).numpy()
    dim = 0
    input_dict = {"input": a, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.logcumsumexp"] = logcumsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logcumsumexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logcumsumexp'.")

check_valid('torch.logcumsumexp', generated_inputs['torch.logcumsumexp'], lib="torch")
