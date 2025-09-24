
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_grad_norm__inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensors
    p1 = torch.randn(3, 4, requires_grad=True)
    p2 = torch.randn(5, 2, requires_grad=True)
    parameters = [p1, p2]
    for p in parameters:
      p.grad = torch.randn_like(p)

    input_dict = {
        "parameters": [p.grad.numpy() for p in parameters],
        "max_norm": 1.0,
        "norm_type": 2.0,
        "error_if_nonfinite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different norm_type and error_if_nonfinite
    p1 = torch.randn(2, 2, requires_grad=True)
    p2 = torch.randn(3, 3, requires_grad=True)
    parameters = [p1, p2]
    for p in parameters:
      p.grad = torch.randn_like(p)
    input_dict = {
        "parameters": [p.grad.numpy() for p in parameters],
        "max_norm": 2.0,
        "norm_type": float('inf'),
        "error_if_nonfinite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single parameter tensor
    p1 = torch.randn(10, requires_grad=True)
    p1.grad = torch.randn_like(p1)

    input_dict = {
        "parameters": [p1.grad.numpy()],
        "max_norm": 0.5,
        "norm_type": 1.0,
        "error_if_nonfinite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Zero norm
    p1 = torch.randn(2, 3, requires_grad=True)
    p2 = torch.randn(4, 5, requires_grad=True)
    parameters = [p1, p2]
    for p in parameters:
      p.grad = torch.randn_like(p)
    input_dict = {
        "parameters": [p.grad.numpy() for p in parameters],
        "max_norm": 0.0,
        "norm_type": 2.0,
        "error_if_nonfinite": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes and values
    p1 = torch.randn(1, 1, 1, requires_grad=True)
    p2 = torch.randn(100, requires_grad=True)
    parameters = [p1, p2]
    for p in parameters:
      p.grad = torch.randn_like(p)
    input_dict = {
        "parameters": [p.grad.numpy() for p in parameters],
        "max_norm": 1.5,
        "norm_type": 1.5,
        "error_if_nonfinite": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.utils.clip_grad_norm__1"] = clip_grad_norm__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.clip_grad_norm__1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.clip_grad_norm__1'.")

check_valid('torch.nn.utils.clip_grad_norm_', generated_inputs['torch.nn.utils.clip_grad_norm__1'], lib="torch")
