
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_add_inputs():
    list_of_inputs = []

    input1 = torch.randn(4).numpy()
    other1 = 2.0
    alpha1 = 1.0
    out1 = torch.empty(4).numpy()

    input_dict1 = {
        "input": input1,
        "other": other1,
        "alpha": alpha1,
        "out": out1
    }

    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(4, 1).numpy()
    other2 = -0.5
    alpha2 = 2.5
    out2 = torch.empty(4, 1).numpy()

    input_dict2 = {
        "input": input2,
        "other": other2,
        "alpha": alpha2,
        "out": out2
    }

    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(-5, 5, (2, 3)).float().numpy()
    other3 = 1.0
    alpha3 = 0.5
    out3 = torch.empty(2, 3).numpy()

    input_dict3 = {
        "input": input3,
        "other": other3,
        "alpha": alpha3,
        "out": out3
    }

    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 2, 2).numpy()
    other4 = -2.0
    alpha4 = -1.0
    out4 = torch.empty(2, 2, 2).numpy()

    input_dict4 = {
        "input": input4,
        "other": other4,
        "alpha": alpha4,
        "out": out4
    }

    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1).numpy()
    other5 = 0.0
    alpha5 = 1.0
    out5 = torch.empty(1).numpy()

    input_dict5 = {
        "input": input5,
        "other": other5,
        "alpha": alpha5,
        "out": out5
    }

    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.add_2"] = torch_add_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.add_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.add_2'.")

check_valid('torch.add', generated_inputs['torch.add_2'], lib="torch")
