
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_div_inputs():
    list_of_inputs = []

    input1 = torch.randn(5).numpy()
    other1 = 2.0
    input_dict1 = {
        "input": input1,
        "other": other1,
        "rounding_mode": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(-5, 5, (3, 4)).float().numpy()
    other2 = -1.5
    input_dict2 = {
        "input": input2,
        "other": other2,
        "rounding_mode": "trunc",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(-5, 5, (2, 2), dtype=torch.int32).numpy()
    other3 = 2.0
    input_dict3 = {
        "input": input3,
        "other": other3,
        "rounding_mode": "floor",
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(2, 3, 4).numpy()
    other4 = 0.5
    input_dict4 = {
        "input": input4,
        "other": other4,
        "rounding_mode": None,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 5, 5).numpy()
    other5 = -2.5
    input_dict5 = {
        "input": input5,
        "other": other5,
        "rounding_mode": 'trunc',
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.div_2"] = torch_div_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.div_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.div_2'.")

check_valid('torch.div', generated_inputs['torch.div_2'], lib="torch")
