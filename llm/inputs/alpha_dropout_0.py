
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def alpha_dropout_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3, 4, 5).numpy()
    input_dict1 = {
        "input": input1,
        "p": 0.5,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(1, 5, 5).numpy()
    input_dict2 = {
        "input": input2,
        "p": 0.2,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(10).numpy()
    input_dict3 = {
        "input": input3,
        "p": 0.8,
        "training": True,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, 2, 2, 2).numpy()
    input_dict4 = {
        "input": input4,
        "p": 0.3,
        "training": False,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(3, 4).numpy()
    input_dict5 = {
        "input": input5,
        "p": 0.7,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(1, 1, 1).numpy()
    input_dict6 = {
        "input": input6,
        "p": 0.9,
        "training": False,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(5).numpy()
    input_dict7 = {
        "input": input7,
        "p": 0.1,
        "training": True,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.functional.alpha_dropout"] = alpha_dropout_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.alpha_dropout' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.alpha_dropout'.")

check_valid('torch.nn.functional.alpha_dropout', generated_inputs['torch.nn.functional.alpha_dropout'], lib="torch")
