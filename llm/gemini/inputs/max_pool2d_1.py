
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def max_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor
    input1 = torch.randn(1, 3, 32, 32).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Int tensor with different kernel_size and stride
    input2 = torch.randint(0, 10, (1, 1, 16, 16)).float().numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Input with negative values and different padding
    input3 = torch.randn(2, 5, 28, 28) * -1.0
    input3 = input3.numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": 4,
        "stride": 2,
        "padding": 2,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Single channel input
    input4 = torch.randn(1, 1, 64, 64).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": 5,
        "stride": 3,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger batch size
    input5 = torch.randn(4, 3, 20, 20).numpy()
    input_dict5 = {
        "input": input5,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool2d_1"] = max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.max_pool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool2d_1'.")

check_valid('torch.nn.functional.max_pool2d', generated_inputs['torch.nn.functional.max_pool2d_1'], lib="torch")
