
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_norm_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, default p, dim=None, keepdim=False
    input_tensor = torch.randn(3, 4).numpy()
    input_dict = {"input": input_tensor, "p": "fro", "dim": None, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float tensor, p=1, dim=0, keepdim=True
    input_tensor = torch.randn(2, 3).numpy()
    input_dict = {"input": input_tensor, "p": 1, "dim": [0], "keepdim": True, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Complex tensor, p=2, dim=(0, 1), keepdim=False
    input_tensor = torch.randn(2, 2, dtype=torch.complex64).numpy()
    input_dict = {"input": input_tensor, "p": 2, "dim": [0, 1], "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float tensor, p=inf, dim=1, keepdim=True, dtype=torch.float64
    input_tensor = torch.randn(4, 5).numpy()
    input_dict = {"input": input_tensor, "p": float('inf'), "dim": [1], "keepdim": True, "out": None, "dtype": torch.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float tensor, p=-inf, dim=None
    input_tensor = torch.randn(2, 3, 4).numpy()
    input_dict = {"input": input_tensor, "p": float('-inf'), "dim": None, "keepdim": False, "out": None, "dtype": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.norm_11"] = torch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.norm_11' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.norm_11'.")

check_valid('torch.norm', generated_inputs['torch.norm_11'], lib="torch")
