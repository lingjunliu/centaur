
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np
import torch.nn.functional as F

def lp_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor
    input1 = torch.randn(1, 3, 32, 32).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "ceil_mode": False,
        "p": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor with different stride and padding
    input2 = torch.randint(0, 10, (2, 1, 16, 16)).float().numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "ceil_mode": True,
        "p": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values in input and different kernel size
    input3 = torch.randn(1, 2, 64, 64) * -1.0
    input3 = input3.numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": (4, 4),
        "stride": (2, 2),
        "padding": (1, 1),
        "ceil_mode": False,
        "p": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4:
    input4 = torch.randn(2, 3, 10, 10).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "ceil_mode": True,
        "p": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.lp_pool2d"] = lp_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.lp_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.lp_pool2d'.")

check_valid('torch.nn.functional.lp_pool2d', generated_inputs['torch.nn.functional.lp_pool2d'], lib="torch")
