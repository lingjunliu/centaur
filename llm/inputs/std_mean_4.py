
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def std_mean_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 4, 5).numpy()
    dim1 = (0, 2)
    unbiased1 = True
    keepdim1 = True
    input_dict1 = {
        "input": input1,
        "dim": dim1,
        "unbiased": unbiased1,
        "keepdim": keepdim1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(0, 10, (3, 5, 2)).float().numpy()
    dim2 = (1,)
    unbiased2 = False
    keepdim2 = False
    input_dict2 = {
        "input": input2,
        "dim": dim2,
        "unbiased": unbiased2,
        "keepdim": keepdim2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 1, 1).numpy()
    dim3 = (0, 1, 2)
    unbiased3 = True
    keepdim3 = False
    input_dict3 = {
        "input": input3,
        "dim": dim3,
        "unbiased": unbiased3,
        "keepdim": keepdim3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(4, 4).numpy()
    dim4 = (0,)
    unbiased4 = False
    keepdim4 = True
    input_dict4 = {
        "input": input4,
        "dim": dim4,
        "unbiased": unbiased4,
        "keepdim": keepdim4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    input5 = torch.randn(2, 2, 2).numpy()
    input_dict5 = {
        "input": input5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.std_mean_4"] = std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.std_mean_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_4'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_4'], lib="torch")
