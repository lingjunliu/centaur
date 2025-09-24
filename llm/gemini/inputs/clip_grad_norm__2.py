
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def clip_grad_norm__inputs():
    list_of_inputs = []

    p1 = torch.nn.Parameter(torch.randn(2, 3, requires_grad=True))
    p2 = torch.nn.Parameter(torch.randn(3, 4, requires_grad=True))
    p3 = torch.nn.Parameter(torch.randn(4, 2, requires_grad=True))
    parameters = [p1, p2, p3]
    for p in parameters:
        p.grad = torch.randn_like(p)
    
    input_dict = {
        "parameters": parameters,
        "max_norm": 1.0,
        "norm_type": 2.0,
        "error_if_nonfinite": False
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    p1 = torch.nn.Parameter(torch.randn(2, 3, requires_grad=True))
    p2 = torch.nn.Parameter(torch.randn(3, 4, requires_grad=True))
    parameters = [p1, p2]
    for p in parameters:
        p.grad = torch.randn_like(p) * 10
    
    input_dict = {
        "parameters": parameters,
        "max_norm": 5.0,
        "norm_type": 2.0,
        "error_if_nonfinite": True
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    p1 = torch.nn.Parameter(torch.randn(2, 3, requires_grad=True))
    p2 = torch.nn.Parameter(torch.randn(3, 4, requires_grad=True))
    p3 = torch.nn.Parameter(torch.randn(4, 2, requires_grad=True))
    parameters = [p1, p2, p3]
    for p in parameters:
        p.grad = torch.randn_like(p)
    
    input_dict = {
        "parameters": parameters,
        "max_norm": 0.5,
        "norm_type": np.inf,
        "error_if_nonfinite": False
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    p1 = torch.nn.Parameter(torch.randn(2, 3, requires_grad=True))
    p2 = torch.nn.Parameter(torch.randn(3, 4, requires_grad=True))
    parameters = [p1, p2]
    for p in parameters:
        p.grad = torch.randn_like(p)
    
    input_dict = {
        "parameters": parameters,
        "max_norm": 1.0,
        "norm_type": 1.0,
        "error_if_nonfinite": False
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    p1 = torch.nn.Parameter(torch.randn(2, 3, requires_grad=True))
    p2 = torch.nn.Parameter(torch.randn(3, 4, requires_grad=True))
    p3 = torch.nn.Parameter(torch.randn(4, 2, requires_grad=True))
    parameters = [p1, p2, p3]
    for p in parameters:
        p.grad = torch.randn_like(p)
    
    input_dict = {
        "parameters": parameters,
        "max_norm": 2.0,
        "norm_type": 0.5,
        "error_if_nonfinite": False
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    p1 = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    p2 = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    parameters = [p1, p2]
    for p in parameters:
        p.grad = torch.randn_like(p)
    
    input_dict = {
        "parameters": parameters,
        "max_norm": 1.0,
        "norm_type": 2.0,
        "error_if_nonfinite": False
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["torch.nn.utils.clip_grad_norm__2"] = clip_grad_norm__inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.utils.clip_grad_norm__2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.utils.clip_grad_norm__2'.")

check_valid('torch.nn.utils.clip_grad_norm_', generated_inputs['torch.nn.utils.clip_grad_norm__2'], lib="torch")
