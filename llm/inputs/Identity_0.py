
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def Identity_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float tensor
    input1 = torch.randn(128, 20).numpy()
    input_dict1 = {"args": [], "kwargs": {}, "input": input1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D int tensor with negative values
    input2 = torch.randint(-10, 10, (32, 32, 3)).numpy()
    input_dict2 = {"args": [1, "abc"], "kwargs": {"a": 123, "b": "def"}, "input": input2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D float tensor
    input3 = torch.randn(100).numpy()
    input_dict3 = {"args": [], "kwargs": {}, "input": input3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D complex tensor
    input4 = (torch.randn(2, 3, 4, 5) + 1j * torch.randn(2, 3, 4, 5)).numpy()
    input_dict4 = {"args": [], "kwargs": {}, "input": input4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Empty tensor
    input5 = torch.empty(0).numpy()
    input_dict5 = {"args": [1.0], "kwargs": {"test": 1}, "input": input5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: 5D float tensor
    input6 = torch.randn(2, 4, 6, 8, 10).numpy()
    input_dict6 = {"args": [], "kwargs": {}, "input": input6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.nn.Identity"] = Identity_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Identity'.")

check_valid('torch.nn.Identity', generated_inputs['torch.nn.Identity'], lib="torch")
