
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def torch_var_inputs():
    list_of_inputs = []

    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {
        "input": input1,
        "dim": 1,
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3, 4).numpy()
    input_dict2 = {
        "input": input2,
        "dim": (0, 2),
        "correction": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5, 5).numpy()
    input_dict3 = {
        "input": input3,
        "dim": 0,
        "correction": 2,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(10).numpy()
    input_dict4 = {
        "input": input4,
        "dim": None,
        "correction": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2).numpy()
    out5 = torch.empty(2).numpy()
    input_dict5 = {
        "input": input5,
        "dim": 1,
        "correction": 1,
        "keepdim": False,
        "out": out5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randn(2, 3, 4).numpy()
    input_dict6 = {
        "input": input6,
        "dim": (1),
        "correction": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randn(3,3, dtype=torch.float64).numpy()
    input_dict7 = {
        "input": input7,
        "dim": 0,
        "correction": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.var_1"] = torch_var_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.var_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.var_1'.")

check_valid('torch.var', generated_inputs['torch.var_1'], lib="torch")
