
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
      if p.grad is None:
        p.grad = torch.randn_like(p)
    max_norm = 1.0
    norm_type = 2.0
    error_if_nonfinite = False
    
    input_dict = {
        "parameters": [p.grad.numpy() for p in parameters],
        "max_norm": max_norm,
        "norm_type": norm_type,
        "error_if_nonfinite": error_if_nonfinite
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.clip_grad_norm__1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.clip_grad_norm__1'.")

check_valid('torch.nn.utils.clip_grad_norm_', generated_inputs['torch.nn.utils.clip_grad_norm__1'], lib="torch")
