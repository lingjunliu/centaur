
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def argsort_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D float tensor
    input1 = torch.randn(4, 4).numpy()
    input_dict1 = {
        "input": input1,
        "dim": 1,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 1D int tensor, descending order
    input2 = torch.randint(0, 10, (10,)).numpy()
    input_dict2 = {
        "input": input2,
        "dim": 0,
        "descending": True,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor, different dim
    input3 = torch.randn(2, 3, 5).numpy()
    input_dict3 = {
        "input": input3,
        "dim": 2,
        "descending": False,
        "stable": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with negative values
    input4 = torch.randint(-5, 5, (3, 3)).float().numpy()
    input_dict4 = {
        "input": input4,
        "dim": 1,
        "descending": True,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 1D float tensor
    input5 = torch.randn(5).numpy()
    input_dict5 = {
        "input": input5,
        "dim": 0,
        "descending": False,
        "stable": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.argsort"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.argsort' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.argsort'.")

check_valid('torch.argsort', generated_inputs['torch.argsort'], lib="torch")
