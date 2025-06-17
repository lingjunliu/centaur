
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def softplus_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "beta": 1.0,
        "threshold": 20.0,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "beta": 0.5,
        "threshold": 10.0,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {
        "beta": 2.0,
        "threshold": 30.0,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 1, 1).numpy()
    input_dict4 = {
        "beta": 0.1,
        "threshold": 5.0,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(5).numpy()
    input_dict5 = {
        "beta": 1.5,
        "threshold": 25.0,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.Softplus"] = softplus_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Softplus'.")

check_valid('torch.nn.Softplus', generated_inputs['torch.nn.Softplus'], lib="torch")
