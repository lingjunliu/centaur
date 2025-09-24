
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "p": 2,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor with dim specified
    input_tensor = torch.randint(-5, 5, (2, 3, 4)).float().numpy()
    input_dict = {
        "input": input_tensor,
        "p": 1,
        "dim": [0, 1],
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor
    input_tensor = torch.complex(torch.randn(2, 2), torch.randn(2, 2)).numpy()
    input_dict = {
        "input": input_tensor,
        "p": 'fro',
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": torch.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float tensor with inf norm
    input_tensor = torch.randn(5).numpy()
    input_dict = {
        "input": input_tensor,
        "p": float('inf'),
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float tensor with negative inf norm
    input_tensor = torch.randn(5).numpy()
    input_dict = {
        "input": input_tensor,
        "p": float('-inf'),
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D Float Tensor
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {
        "input": input_tensor,
        "p": 2,
        "dim": (0, 1),
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.norm_13"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_13' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_13'.")

check_valid('torch.norm', generated_inputs['torch.norm_13'], lib="torch")
