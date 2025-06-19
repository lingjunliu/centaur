
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logsumexp_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D tensor, dim=1, keepdim=False
    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "dim": [1],
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 3D tensor, dim=(0, 2), keepdim=True
    input2 = torch.randn(2, 3, 5).numpy()
    input_dict2 = {
        "input": input2,
        "dim": [0, 2],
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor, dim=0, keepdim=False
    input3 = torch.randn(5).numpy()
    input_dict3 = {
        "input": input3,
        "dim": [0],
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with negative values, dim=1, keepdim=True
    input4 = torch.randn(2, 4) * -1.0
    input4 = input4.numpy()
    input_dict4 = {
        "input": input4,
        "dim": [1],
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 4D tensor, dim=(1, 3), keepdim=False
    input5 = torch.randn(2, 3, 4, 5).numpy()
    input_dict5 = {
        "input": input5,
        "dim": [1, 3],
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Integer tensor, dim=0, keepdim=True
    input6 = torch.randint(0, 10, (3, 4)).float().numpy()
    input_dict6 = {
        "input": input6,
        "dim": [0],
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs["torch.logsumexp_5"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logsumexp_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp_5'.")

check_valid('torch.logsumexp', generated_inputs['torch.logsumexp_5'], lib="torch")
