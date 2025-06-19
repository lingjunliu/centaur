
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input_dict = {
        "input": torch.randn(3, 4).numpy(),
        "p": 2.0,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor with dim specified
    input_dict = {
        "input": torch.randint(-5, 5, (2, 3, 4)).float().numpy(),
        "p": 1.0,
        "dim": 1,
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor
    input_dict = {
        "input": (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy(),
        "p": 'fro',
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different p value (inf)
    input_dict = {
        "input": torch.randn(5).numpy(),
        "p": float('inf'),
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple dimensions in dim
    input_dict = {
        "input": torch.randn(2, 3, 4).numpy(),
        "p": 2.0,
        "dim": (0, 1),
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.norm_7"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_7'.")

check_valid('torch.norm', generated_inputs['torch.norm_7'], lib="torch")
