
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_var_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D float tensor, dim=1, keepdim=True
    input = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input,
        "dim": 1,
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: 3D float tensor, dim=(0, 2), keepdim=False, correction=0
    input = torch.randn(2, 3, 5).numpy()
    input_dict = {
        "input": input,
        "dim": (0, 2),
        "correction": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: 1D float tensor with negative values, dim=None, keepdim=False
    input = torch.randn(6).numpy() * -1
    input_dict = {
        "input": input,
        "dim": None,
        "correction": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: 4D complex tensor, dim=3, keepdim=True, correction=2
    input = (torch.randn(2, 2, 3, 4) + 1j * torch.randn(2, 2, 3, 4)).numpy()
    input_dict = {
        "input": input,
        "dim": 3,
        "correction": 2,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: 2D tensor, no dim specified, correction=1, keepdim=False
    input = torch.randn(5, 5).numpy()
    input_dict = {
        "input": input,
        "dim": None,
        "correction": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.var_4"] = torch_var_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_4'.")

check_valid('torch.var', generated_inputs['torch.var_4'], lib="torch")
