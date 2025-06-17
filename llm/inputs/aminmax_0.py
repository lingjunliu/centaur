
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def aminmax_inputs():
    list_of_inputs = []

    input1 = torch.tensor([1, -3, 5]).numpy()
    input_dict1 = {
        "input": input1,
        "dim": None,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.tensor([1, -3, 5, float('nan')]).numpy()
    input_dict2 = {
        "input": input2,
        "dim": None,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.arange(10).view(2, 5).numpy()
    input_dict3 = {
        "input": input3,
        "dim": 0,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(3, 4, 5).numpy()
    input_dict4 = {
        "input": input4,
        "dim": 1,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randint(0, 10, (2, 2, 2)).float().numpy()
    input_dict5 = {
        "input": input5,
        "dim": 2,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(2, 3, 4).double().numpy()
    input_dict6 = {
        "input": input6,
        "dim": None,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randint(-5, 5, (5,)).int().numpy()
    input_dict7 = {
        "input": input7,
        "dim": None,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = torch.tensor([[-1, 2], [-3, 4]], dtype=torch.float64).numpy()
    input_dict8 = {
        "input": input8,
        "dim": 0,
        "keepdim": False,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input9 = torch.tensor([[-1, 2], [-3, 4]], dtype=torch.float64).numpy()
    input_dict9 = {
        "input": input9,
        "dim": 1,
        "keepdim": True,
        "out": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))
    
    return list_of_inputs

generated_inputs["torch.aminmax"] = aminmax_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.aminmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.aminmax'.")

check_valid('torch.aminmax', generated_inputs['torch.aminmax'], lib="torch")
