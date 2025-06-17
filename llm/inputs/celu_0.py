
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def celu_inputs():
    list_of_inputs = []

    input1 = torch.randn(3, 4, 5).numpy()
    input_dict1 = {
        "input": input1,
        "alpha": 1.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 2).numpy()
    input_dict2 = {
        "input": input2,
        "alpha": 0.5,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 5, 5).numpy()
    input_dict3 = {
        "input": input3,
        "alpha": 2.0,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    input4 = torch.randn(1, 1, 7, 7).numpy()
    input_dict4 = {
        "input": input4,
        "alpha": 1.5,
        "inplace": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(size=(4,)).numpy()
    input_dict5 = {
        "input": input5,
        "alpha": 0.75,
        "inplace": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.functional.celu"] = celu_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.celu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.celu'.")

check_valid('torch.nn.functional.celu', generated_inputs['torch.nn.functional.celu'], lib="torch")
