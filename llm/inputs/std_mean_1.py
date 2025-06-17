
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def std_mean_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensor, unbiased=True, keepdim=False
    input1 = torch.randn(2, 3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "unbiased": True,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor, unbiased=False, keepdim=True
    input2 = torch.randint(0, 10, (5, 5)).numpy()
    input_dict2 = {
        "input": input2,
        "unbiased": False,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 1D tensor, unbiased=True, keepdim=True
    input3 = torch.randn(10).numpy()
    input_dict3 = {
        "input": input3,
        "unbiased": True,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensor with negative values, unbiased=False, keepdim=False
    input4 = torch.randint(-10, 10, (3, 3)).float().numpy()
    input_dict4 = {
        "input": input4,
        "unbiased": False,
        "keepdim": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger tensor, unbiased=True, keepdim=True
    input5 = torch.randn(4, 4, 4, 4).numpy()
    input_dict5 = {
        "input": input5,
        "unbiased": True,
        "keepdim": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    

    return list_of_inputs

generated_inputs["torch.std_mean_1"] = std_mean_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.std_mean_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.std_mean_1'.")

check_valid('torch.std_mean', generated_inputs['torch.std_mean_1'], lib="torch")
