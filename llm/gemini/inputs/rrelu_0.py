
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def rrelu_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4, 5).numpy()
    input_dict1 = {
        "input": input1,
        "lower": 1. / 8,
        "upper": 1. / 3,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 2, 2, 2).numpy()
    input_dict2 = {
        "input": input2,
        "lower": 0.1,
        "upper": 0.4,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 5).numpy()
    input_dict3 = {
        "input": input3,
        "lower": 0.0,
        "upper": 1.0,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(10).numpy() * -1
    input_dict4 = {
        "input": input4,
        "lower": 1. / 16,
        "upper": 1. / 4,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randint(-5, 5, (2, 3)).float().numpy()
    input_dict5 = {
        "input": input5,
        "lower": 0.2,
        "upper": 0.3,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(1).numpy()
    input_dict6 = {
        "input": input6,
        "lower": 1. / 8,
        "upper": 1. / 3,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    return list_of_inputs

generated_inputs["torch.nn.functional.rrelu"] = rrelu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.rrelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.rrelu'.")

check_valid('torch.nn.functional.rrelu', generated_inputs['torch.nn.functional.rrelu'], lib="torch")
