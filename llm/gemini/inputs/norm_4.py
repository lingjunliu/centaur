
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "p": 2.0,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor with dim
    input2 = torch.randint(-5, 5, (2, 3, 4)).float().numpy()
    input_dict2 = {
        "input": input2,
        "p": 1.0,
        "dim": (0, 1),
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Complex tensor with different p
    input3 = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict3 = {
        "input": input3,
        "p": float('inf'),
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Negative values, p='fro'
    input4 = (torch.randn(5, 5) - 2).numpy()
    input_dict4 = {
        "input": input4,
        "p": 'fro',
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D tensor
    input5 = torch.arange(-5, 5, dtype=torch.float).numpy()
    input_dict5 = {
        "input": input5,
        "p": 2.0,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    return list_of_inputs

generated_inputs["torch.norm_4"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_4'.")

check_valid('torch.norm', generated_inputs['torch.norm_4'], lib="torch")
