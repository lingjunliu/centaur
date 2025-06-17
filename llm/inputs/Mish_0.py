
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def mish_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "input": input1,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "input": input2,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 2, 3, 4).numpy()
    input_dict3 = {
        "input": input3,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(5).numpy()
    input_dict4 = {
        "input": input4,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2, 2, 2, 2).numpy()
    input_dict5 = {
        "input": input5,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(2).float().numpy()
    input_dict6 = {
        "input": input6,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    
    input8 = torch.randn(2).double().numpy()
    input_dict8 = {
        "input": input8,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    return list_of_inputs

generated_inputs["torch.nn.Mish"] = mish_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Mish' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Mish'.")

check_valid('torch.nn.Mish', generated_inputs['torch.nn.Mish'], lib="torch")
