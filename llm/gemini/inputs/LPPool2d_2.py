
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def LPPool2d_inputs():
    list_of_inputs = []

    input1 = torch.randn(20, 16, 50, 32).numpy()
    input_dict1 = {
        "norm_type": 2.0,
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "ceil_mode": False,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input2 = torch.randn(20, 16, 50, 32).numpy()
    input_dict2 = {
        "norm_type": 1.2,
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "ceil_mode": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input3 = torch.randn(5, 3, 25, 25).numpy()
    input_dict3 = {
        "norm_type": 1.0,
        "kernel_size": (5, 5),
        "stride": (5, 5),
        "ceil_mode": False,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input4 = torch.randn(1, 1, 100, 100).numpy()
    input_dict4 = {
        "norm_type": 2.0,
        "kernel_size": (10, 10),
        "stride": (10, 10),
        "ceil_mode": True,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input5 = torch.randn(1, 1, 50, 50).numpy()
    input_dict5 = {
        "norm_type": 0.5,
        "kernel_size": (7, 7),
        "stride": (1, 1),
        "ceil_mode": False,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LPPool2d_2"] = LPPool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.LPPool2d', generated_inputs['torch.nn.LPPool2d_2'], lib="torch")
