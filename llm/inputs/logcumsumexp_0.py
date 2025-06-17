
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logcumsumexp_inputs():
    list_of_inputs = []

    # Example 1: 1D tensor
    a = torch.randn(10).numpy()
    dim = 0
    input_dict = {"input": a, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: 2D tensor, dim=0
    a = torch.randn(5, 5).numpy()
    dim = 0
    input_dict = {"input": a, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: 2D tensor, dim=1
    a = torch.randn(5, 5).numpy()
    dim = 1
    input_dict = {"input": a, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: 3D tensor, dim=0
    a = torch.randn(3, 4, 5).numpy()
    dim = 0
    input_dict = {"input": a, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: 3D tensor, dim=1
    a = torch.randn(3, 4, 5).numpy()
    dim = 1
    input_dict = {"input": a, "dim": dim}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.logcumsumexp"] = logcumsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logcumsumexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logcumsumexp'.")

check_valid('torch.logcumsumexp', generated_inputs['torch.logcumsumexp'], lib="torch")
