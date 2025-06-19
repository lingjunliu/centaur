
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def hardsigmoid_inputs():
    list_of_inputs = []

    input_1 = torch.randn(2).numpy()
    input_dict_1 = {
        "inplace": False,
        "input": input_1
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    input_2 = torch.randn(2, 3).numpy()
    input_dict_2 = {
        "inplace": True,
        "input": input_2
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    input_3 = torch.randn(2, 3, 4).numpy()
    input_dict_3 = {
        "inplace": False,
        "input": input_3
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    input_4 = torch.randint(-5, 5, (5,)).float().numpy()
    input_dict_4 = {
        "inplace": True,
        "input": input_4
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    input_5 = torch.randint(-5, 5, (2, 2, 2)).float().numpy()
    input_dict_5 = {
        "inplace": False,
        "input": input_5
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    input_6 = torch.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).numpy()
    input_dict_6 = {
        "inplace": True,
        "input": input_6
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    input_7 = torch.randn(1, 1, 1, 1).numpy()
    input_dict_7 = {
        "inplace": False,
        "input": input_7
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    return list_of_inputs

generated_inputs["torch.nn.Hardsigmoid"] = hardsigmoid_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Hardsigmoid' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Hardsigmoid'.")

check_valid('torch.nn.Hardsigmoid', generated_inputs['torch.nn.Hardsigmoid'], lib="torch")
