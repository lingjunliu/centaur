
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def Hardtanh_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "min_val": -2.0,
        "max_val": 2.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "min_val": -0.5,
        "max_val": 1.5,
        "inplace": True,
        "min_value": None,
        "max_value": None,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(1, 2, 3, 4).numpy()
    input_dict3 = {
        "min_val": -1.0,
        "max_val": 1.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(5).numpy()
    input_dict4 = {
        "min_val": -5.0,
        "max_val": 0.0,
        "inplace": True,
        "min_value": None,
        "max_value": None,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(2, 2, 2).numpy()
    input_dict5 = {
        "min_val": 0.0,
        "max_val": 5.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    input6 = torch.randint(-10, 10, (3,4)).float().numpy()
    input_dict6 = {
        "min_val": -3.0,
        "max_val": 7.0,
        "inplace": False,
        "min_value": None,
        "max_value": None,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.Hardtanh"] = Hardtanh_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.Hardtanh', generated_inputs['torch.nn.Hardtanh'], lib="torch")
