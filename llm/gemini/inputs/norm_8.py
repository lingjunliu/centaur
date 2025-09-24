
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensor
    input_dict = {
        "input": torch.randn(3, 3).numpy(),
        "p": 2,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Integer tensor with specified dimension - CAST TO FLOAT
    input_dict = {
        "input": torch.randint(-5, 5, (2, 3, 4)).float().numpy(),
        "p": 1,
        "dim": 1,
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Complex tensor
    input_dict = {
        "input": torch.randn(2, 2, dtype=torch.complex64).numpy(),
        "p": 'fro',
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Float tensor with inf norm
    input_dict = {
        "input": torch.randn(4, 5).numpy(),
        "p": float('inf'),
        "dim": 0,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 5: Float tensor with negative inf norm
    input_dict = {
        "input": torch.randn(4, 5).numpy(),
        "p": float('-inf'),
        "dim": 1,
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 6: Float tensor with specific dtype
    input_dict = {
        "input": torch.randn(2, 3).numpy(),
        "p": 2,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 7: Multiple dimensions
    input_dict = {
        "input": torch.randn(2, 2, 2).numpy(),
        "p": 2,
        "dim": (0,1),
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 8: p = 0
    input_dict = {
        "input": torch.randn(2, 2).numpy(),
        "p": 0,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.norm_8"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_8'.")

check_valid('torch.norm', generated_inputs['torch.norm_8'], lib="torch")
