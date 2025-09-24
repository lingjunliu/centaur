
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def ReLU_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 2, 3, 4).numpy()
    input_dict4 = {
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5).numpy()
    input_dict5 = {
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randint(-5, 5, (2, 2)).float().numpy()
    input_dict6 = {
        "inplace": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(2, 2).double().numpy()
    input_dict7 = {
        "inplace": False,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.ReLU"] = ReLU_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ReLU' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReLU'.")

check_valid('torch.nn.ReLU', generated_inputs['torch.nn.ReLU'], lib="torch")
