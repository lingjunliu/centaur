
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def gelu_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4).numpy()
    input_dict1 = {
        "input": input1,
        "approximate": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3, 4).numpy()
    input_dict2 = {
        "input": input2,
        "approximate": 'tanh'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 2, 3, 4).numpy()
    input_dict3 = {
        "input": input3,
        "approximate": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(5).numpy()
    input_dict4 = {
        "input": input4,
        "approximate": 'tanh'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(10, 10).numpy()
    input_dict5 = {
        "input": input5,
        "approximate": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 2, 2, 2, 2).numpy()
    input_dict6 = {
        "input": input6,
        "approximate": 'tanh'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    input7 = torch.randint(-10, 10, (3, 4)).float().numpy()
    input_dict7 = {
        "input": input7,
        "approximate": 'none'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.functional.gelu"] = gelu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.gelu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.gelu'.")

check_valid('torch.nn.functional.gelu', generated_inputs['torch.nn.functional.gelu'], lib="torch")
