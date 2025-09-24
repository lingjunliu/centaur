
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def baddbmm_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    input_tensor = torch.randn(10, 3, 5).numpy()
    batch1 = torch.randn(10, 3, 4).numpy()
    batch2 = torch.randn(10, 4, 5).numpy()
    input_dict = {
        "input": input_tensor,
        "batch1": batch1,
        "batch2": batch2,
        "beta": 1.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors with different alpha and beta
    input_tensor = torch.randint(0, 10, (5, 2, 3)).float().numpy()
    batch1 = torch.randint(0, 10, (5, 2, 4)).float().numpy()
    batch2 = torch.randint(0, 10, (5, 4, 3)).float().numpy()
    input_dict = {
        "input": input_tensor,
        "batch1": batch1,
        "batch2": batch2,
        "beta": 0.5,
        "alpha": 2.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values and beta = 0
    input_tensor = torch.randn(2, 4, 4).numpy() * -1
    batch1 = torch.randn(2, 4, 2).numpy() * -1
    batch2 = torch.randn(2, 2, 4).numpy() * -1
    input_dict = {
        "input": input_tensor,
        "batch1": batch1,
        "batch2": batch2,
        "beta": 0.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different batch size
    input_tensor = torch.randn(3, 5, 7).numpy()
    batch1 = torch.randn(3, 5, 4).numpy()
    batch2 = torch.randn(3, 4, 7).numpy()
    input_dict = {
        "input": input_tensor,
        "batch1": batch1,
        "batch2": batch2,
        "beta": 1.0,
        "alpha": 0.5,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Providing an output tensor
    input_tensor = torch.randn(4, 2, 6).numpy()
    batch1 = torch.randn(4, 2, 3).numpy()
    batch2 = torch.randn(4, 3, 6).numpy()
    out_tensor = torch.empty(4, 2, 6).numpy()
    input_dict = {
        "input": input_tensor,
        "batch1": batch1,
        "batch2": batch2,
        "beta": 1.0,
        "alpha": 1.0,
        "out": out_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.baddbmm"] = baddbmm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.baddbmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.baddbmm'.")

check_valid('torch.baddbmm', generated_inputs['torch.baddbmm'], lib="torch")
