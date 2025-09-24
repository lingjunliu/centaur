
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def torch_max_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor, dim=1, keepdim=False
    input1 = torch.randn(4, 4).numpy()
    input_dict1 = {
        "input": input1,
        "dim": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor, dim=0, keepdim=True
    input2 = torch.randn(2, 3, 5).numpy()
    input_dict2 = {
        "input": input2,
        "dim": 0,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 2D tensor with negative values, dim=1, keepdim=True
    input3 = torch.randn(3, 3) * -1
    input3 = input3.numpy()
    input_dict3 = {
        "input": input3,
        "dim": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D tensor, dim=0, keepdim=False. Note: dim=0 is technically correct for 1D, but has no effect
    input4 = torch.randn(5).numpy()
    input_dict4 = {
        "input": input4,
        "dim": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D Integer tensor, dim=0, keepdim=False
    input5 = torch.randint(0, 10, (2, 5)).numpy()
    input_dict5 = {
        "input": input5,
        "dim": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))


    return list_of_inputs

generated_inputs["torch.max_2"] = torch_max_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.max_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.max_2'.")

check_valid('torch.max', generated_inputs['torch.max_2'], lib="torch")
