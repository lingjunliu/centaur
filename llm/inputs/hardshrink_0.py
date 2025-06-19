
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def hardshrink_inputs():
    list_of_inputs = []

    input1 = torch.randn(5).numpy()
    input_dict1 = {
        "input": input1,
        "lambd": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "input": input2,
        "lambd": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(-5, 5, (3, 4)).float().numpy()
    input_dict3 = {
        "input": input3,
        "lambd": 0.75
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, 2).numpy()
    input_dict4 = {
        "input": input4,
        "lambd": 0.2
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 5, 5).numpy()
    input_dict5 = {
        "input": input5,
        "lambd": 0.9
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(size=(4, 4)).numpy()
    input_dict6 = {
        "input": input6,
        "lambd": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randn(size=(1,2,3,4)).numpy()
    input_dict7 = {
        "input": input7,
        "lambd": 0.6
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.functional.hardshrink"] = hardshrink_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.hardshrink' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.hardshrink'.")

check_valid('torch.nn.functional.hardshrink', generated_inputs['torch.nn.functional.hardshrink'], lib="torch")
