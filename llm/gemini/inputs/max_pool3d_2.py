
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def max_pool3d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor
    input1 = torch.randn(1, 3, 10, 10, 10).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor
    input2 = torch.randint(0, 10, (2, 4, 8, 8, 8)).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values, different kernel size, stride and padding
    input3 = torch.randn(1, 1, 12, 12, 12) * -1
    input3 = input3.numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": (5, 5, 5),
        "stride": (3, 3, 3),
        "padding": (2, 2, 2),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different dilation
    input4 = torch.randn(2, 2, 15, 15, 15).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": (2, 2, 2),
        "return_indices": True,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Asymmetric kernel size, stride and padding
    input5 = torch.randn(1, 3, 20, 20, 20).numpy()
    input_dict5 = {
        "input": input5,
        "kernel_size": (2, 3, 4),
        "stride": (1, 2, 3),
        "padding": (0, 1, 2),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool3d_2"] = max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.max_pool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool3d_2'.")

check_valid('torch.nn.functional.max_pool3d', generated_inputs['torch.nn.functional.max_pool3d_2'], lib="torch")
