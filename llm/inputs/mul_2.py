
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_mul_inputs():
    list_of_inputs = []

    input1 = torch.randn(3).numpy()
    other1 = 2.0
    out1 = torch.empty(3).numpy()

    input_dict1 = {
        "input": input1,
        "other": other1,
        "out": out1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randint(0, 10, (2, 3)).float().numpy()
    other2 = -0.5
    out2 = torch.empty(2, 3).numpy()
    input_dict2 = {
        "input": input2,
        "other": other2,
        "out": out2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 4, 4).numpy()
    other3 = 1.5
    out3 = torch.empty(1, 4, 4).numpy()
    input_dict3 = {
        "input": input3,
        "other": other3,
        "out": out3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randint(-5, 5, (5,)).float().numpy()
    other4 = 0.0
    out4 = torch.empty(5).numpy()
    input_dict4 = {
        "input": input4,
        "other": other4,
        "out": out4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2, 2).numpy()
    other5 = -2.5
    out5 = torch.empty(2, 2, 2).numpy()
    input_dict5 = {
        "input": input5,
        "other": other5,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.mul_2"] = torch_mul_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.mul_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.mul_2'.")

check_valid('torch.mul', generated_inputs['torch.mul_2'], lib="torch")
