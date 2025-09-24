
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_norm_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensor, default parameters
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor, "p": 'fro', "dim": None, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Float tensor, p=1, dim=0, keepdim=True
    input_tensor = torch.randn(2, 3).numpy()
    input_dict = {"input": input_tensor, "p": 1.0, "dim": [0], "keepdim": True, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Complex tensor, p=2, dim=1
    input_tensor = torch.complex(torch.randn(2, 2), torch.randn(2, 2)).numpy()
    input_dict = {"input": input_tensor, "p": 2.0, "dim": [1], "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: 3D float tensor, p=inf, dim=(0, 2)
    input_tensor = torch.randn(2, 3, 2).numpy()
    input_dict = {"input": input_tensor, "p": float('inf'), "dim": [0, 2], "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: 1D float tensor, p=-inf
    input_tensor = torch.randn(5).numpy()
    input_dict = {"input": input_tensor, "p": float('-inf'), "dim": None, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.norm_14"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_14' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_14'.")

check_valid('torch.norm', generated_inputs['torch.norm_14'], lib="torch")
