
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def addbmm_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors
    M = torch.randn(3, 5).numpy()
    batch1 = torch.randn(10, 3, 4).numpy()
    batch2 = torch.randn(10, 4, 5).numpy()
    input_dict = {
        "input": M,
        "batch1": batch1,
        "batch2": batch2,
        "beta": 1.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Integer tensors
    M = torch.randint(0, 10, (3, 5)).numpy()
    batch1 = torch.randint(0, 10, (10, 3, 4)).numpy()
    batch2 = torch.randint(0, 10, (10, 4, 5)).numpy()
    input_dict = {
        "input": M,
        "batch1": batch1,
        "batch2": batch2,
        "beta": 2,
        "alpha": 3,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Negative values and different beta/alpha
    M = torch.randn(3, 5).numpy() * -1
    batch1 = torch.randn(5, 3, 4).numpy()
    batch2 = torch.randn(5, 4, 5).numpy()
    input_dict = {
        "input": M,
        "batch1": batch1,
        "batch2": batch2,
        "beta": 0.5,
        "alpha": 1.5,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: beta = 0
    M = torch.randn(3, 5).numpy()
    batch1 = torch.randn(2, 3, 4).numpy()
    batch2 = torch.randn(2, 4, 5).numpy()
    input_dict = {
        "input": M,
        "batch1": batch1,
        "batch2": batch2,
        "beta": 0.0,
        "alpha": 1.0,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Specifying 'out'
    M = torch.randn(3, 5).numpy()
    batch1 = torch.randn(4, 3, 4).numpy()
    batch2 = torch.randn(4, 4, 5).numpy()
    out = torch.empty(3, 5).numpy()
    input_dict = {
        "input": M,
        "batch1": batch1,
        "batch2": batch2,
        "beta": 1.0,
        "alpha": 1.0,
        "out": out
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.addbmm"] = addbmm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.addbmm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.addbmm'.")

check_valid('torch.addbmm', generated_inputs['torch.addbmm'], lib="torch")
