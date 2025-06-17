
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_cholesky_inputs():
    list_of_inputs = []

    # Input 1: Basic symmetric positive-definite matrix
    a = torch.randn(3, 3)
    a = a @ a.mT + torch.eye(3) * 1e-3
    input_dict = {"input": a.numpy(), "upper": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batched input
    a = torch.randn(2, 2, 2)
    a = a @ a.mT + torch.eye(2) * 1e-3
    input_dict = {"input": a.numpy(), "upper": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Larger matrix
    a = torch.randn(5, 5)
    a = a @ a.mT + torch.eye(5) * 1e-3
    input_dict = {"input": a.numpy(), "upper": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Another batched input with different dimensions
    a = torch.randn(4, 3, 3)
    a = a @ a.mT + torch.eye(3) * 1e-3
    input_dict = {"input": a.numpy(), "upper": True}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5: Float64 tensor
    a = torch.randn(3, 3, dtype=torch.float64)
    a = a @ a.mT + torch.eye(3, dtype=torch.float64) * 1e-6
    input_dict = {"input": a.numpy(), "upper": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.cholesky"] = torch_cholesky_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.cholesky' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.cholesky'.")

check_valid('torch.cholesky', generated_inputs['torch.cholesky'], lib="torch")
