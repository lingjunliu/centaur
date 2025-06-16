
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def ReLU6_inputs():
    list_of_inputs = []

    input1 = torch.randn(2).numpy()
    input_dict1 = {
        "inplace": False,
        "input": input1
    }
    list_of_inputs.append(input_dict1)

    input2 = torch.randn(2, 3).numpy()
    input_dict2 = {
        "inplace": True,
        "input": input2
    }
    list_of_inputs.append(input_dict2)

    input3 = torch.randn(2, 3, 4).numpy()
    input_dict3 = {
        "inplace": False,
        "input": input3
    }
    list_of_inputs.append(input_dict3)

    input4 = torch.randint(-5, 10, (2, 3, 4, 5)).numpy()
    input_dict4 = {
        "inplace": True,
        "input": input4
    }
    list_of_inputs.append(input_dict4)

    input5 = torch.randn(1, 5, 5).numpy()
    input_dict5 = {
        "inplace": False,
        "input": input5
    }
    list_of_inputs.append(input_dict5)
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ReLU6"] = ReLU6_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.ReLU6', generated_inputs['torch.nn.ReLU6'], lib="torch")
