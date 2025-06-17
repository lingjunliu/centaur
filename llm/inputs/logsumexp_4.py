
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def logsumexp_inputs():
    list_of_inputs = []

    # Example 1: Basic 2D tensor, dim=1, keepdim=False
    input_1 = torch.randn(3, 3).numpy()
    input_dict_1 = {
        "input": input_1,
        "dim": (1,),
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Example 2: 3D tensor, dim=(0, 2), keepdim=True
    input_2 = torch.randn(2, 4, 3).numpy()
    input_dict_2 = {
        "input": input_2,
        "dim": (0, 2),
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Example 3: 1D tensor, dim=0, keepdim=False
    input_3 = torch.randn(5).numpy()
    input_dict_3 = {
        "input": input_3,
        "dim": (0,),
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Example 4: 2D tensor with negative values, dim=0, keepdim=True
    input_4 = torch.randn(2, 5).numpy() * -1
    input_dict_4 = {
        "input": input_4,
        "dim": (0,),
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Example 5: 4D tensor, dim=(1, 3), keepdim=False
    input_5 = torch.randn(2, 3, 4, 5).numpy()
    input_dict_5 = {
        "input": input_5,
        "dim": (1, 3),
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Example 6: Float64 tensor
    input_6 = torch.randn(3, 2, dtype=torch.float64).numpy()
    input_dict_6 = {
        "input": input_6,
        "dim": (0,),
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Example 7: Tensor with some zeros
    input_7 = torch.randn(2, 2).numpy()
    input_7[0, 0] = 0
    input_dict_7 = {
        "input": input_7,
        "dim": (1,),
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    return list_of_inputs

generated_inputs["torch.logsumexp_4"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logsumexp_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp_4'.")

check_valid('torch.logsumexp', generated_inputs['torch.logsumexp_4'], lib="torch")
