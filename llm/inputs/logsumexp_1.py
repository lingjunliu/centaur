
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def logsumexp_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 3).numpy()
    dim1 = 1
    keepdim1 = False
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "keepdim": keepdim1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 4, 5).numpy()
    dim2 = 0
    keepdim2 = True
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "keepdim": keepdim2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 5, 2, 3).numpy()
    dim3 = 1
    keepdim3 = False
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "keepdim": keepdim3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(4, 4).numpy()
    dim4 = 0
    keepdim4 = False
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "keepdim": keepdim4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 3, 4).numpy()
    dim5 = 2
    keepdim5 = True
    input_dict5 = {
        "input": input5,
        "dim": dim5,
        "keepdim": keepdim5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.logsumexp_1"] = logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.logsumexp_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.logsumexp_1'.")

check_valid('torch.logsumexp', generated_inputs['torch.logsumexp_1'], lib="torch")
