
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def AvgPool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with N, C, L
    input1 = torch.randn(2, 3, 10).numpy()
    input_dict1 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2:  C, L input
    input2 = torch.randn(3, 10).numpy()
    input_dict2 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: ceil_mode = True
    input3 = torch.randn(2, 3, 10).numpy()
    input_dict3 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": True,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: count_include_pad = False
    input4 = torch.randn(2, 3, 10).numpy()
    input_dict4 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different kernel_size, stride, padding
    input5 = torch.randn(2, 3, 15).numpy()
    input_dict5 = {
        "kernel_size": 4,
        "stride": 3,
        "padding": 2,
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AvgPool1d_1"] = AvgPool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AvgPool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool1d_1'.")

check_valid('torch.nn.AvgPool1d', generated_inputs['torch.nn.AvgPool1d_1'], lib="torch")
