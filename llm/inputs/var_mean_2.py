
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def var_mean_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor, unbiased=True, dim=None
    input1 = torch.randn(5).numpy()
    input_dict1 = {
        "input": input1,
        "dim": None,
        "unbiased": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensor, unbiased=False, keepdim=True, dim=0
    input2 = torch.randint(-5, 5, (3, 4)).numpy()
    input_dict2 = {
        "input": input2,
        "dim": (0,),
        "unbiased": False,
        "keepdim": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D float tensor, unbiased=True, keepdim=False, dim=[1, 2]
    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {
        "input": input3,
        "dim": (1, 2),
        "unbiased": True,
        "keepdim": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D float tensor, unbiased=False, keepdim=True, dim=[0, 2, 3]
    input4 = torch.randn(2, 2, 2, 2).numpy()
    input_dict4 = {
        "input": input4,
        "dim": (0, 2, 3),
        "unbiased": False,
        "keepdim": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 5D float tensor with negative values, unbiased=True, keepdim=False, dim=[0, 1, 2, 3]
    input5 = torch.randn(2, 2, 2, 2, 2).numpy()
    input_dict5 = {
        "input": input5,
        "dim": (0, 1, 2, 3),
        "unbiased": True,
        "keepdim": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    

    return list_of_inputs

generated_inputs["torch.var_mean_2"] = var_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_mean_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_mean_2'.")

check_valid('torch.var_mean', generated_inputs['torch.var_mean_2'], lib="torch")
