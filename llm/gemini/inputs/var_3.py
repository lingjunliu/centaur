
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_var_inputs():
    list_of_inputs = []

    # Test case 1: Basic 2D tensor, dim=1, keepdim=True
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": [1],
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: 3D tensor, dim=(0, 2), keepdim=False
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": [0, 2],
        "correction": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: 1D tensor, no dim, correction=0
    input_tensor = torch.arange(1, 6, dtype=torch.float).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": None,
        "correction": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Tensor with negative values, dim=0
    input_tensor = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]]).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": [0],
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: Larger tensor, dim=(1, 2), keepdim=True, correction=2
    input_tensor = torch.randn(3, 4, 5, 2).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": [1, 2],
        "correction": 2,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.var_3"] = torch_var_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_3'.")

check_valid('torch.var', generated_inputs['torch.var_3'], lib="torch")
