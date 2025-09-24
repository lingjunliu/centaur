
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def LeakyReLU_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "negative_slope": 0.1,
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "negative_slope": 0.01,
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randint(-5, 5, (2, 3, 4)).float().numpy()
    input_dict3 = {
        "negative_slope": 0.2,
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 5, 5).numpy()
    input_dict4 = {
        "negative_slope": 0.0,
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 8, 10, 10).numpy()
    input_dict5 = {
        "negative_slope": 0.5,
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input6 = torch.randn(size=(3,4,5), dtype=torch.float64).numpy()
    input_dict6 = {
        "negative_slope": 0.3,
        "inplace": True,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input7 = torch.randint(-10, 10, (4,4), dtype=torch.int32).float().numpy()
    input_dict7 = {
        "negative_slope": 0.05,
        "inplace": False,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs["torch.nn.LeakyReLU"] = LeakyReLU_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.LeakyReLU', generated_inputs['torch.nn.LeakyReLU'], lib="torch")
