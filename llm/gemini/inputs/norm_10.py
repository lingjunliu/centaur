
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

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

    input2 = torch.randn(2, 3, 4).numpy()
    input_dict2 = {
        "input": input2,
        "p": 'fro',
        "dim": (0, 1),
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5).numpy()
    input_dict3 = {
        "input": input3,
        "p": float('inf'),
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = (torch.randn(2, 2) + 1j * torch.randn(2, 2)).numpy()
    input_dict4 = {
        "input": input4,
        "p": 2.0,
        "dim": None,
        "keepdim": False,
        "out": None,
        "dtype": torch.complex128
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(4, 4).numpy()
    input_dict5 = {
        "input": input5,
        "p": 1.0,
        "dim": 0,
        "keepdim": True,
        "out": None,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.norm_10"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_10' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_10'.")

check_valid('torch.norm', generated_inputs['torch.norm_10'], lib="torch")
