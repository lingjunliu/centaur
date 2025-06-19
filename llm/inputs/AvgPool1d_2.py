
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import numpy as np
import copy

def torch_nn_AvgPool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with N, C, L
    input1 = torch.randn(2, 3, 10).numpy()
    input_dict1 = {
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (1,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: C, L input
    input2 = torch.randn(3, 15).numpy()
    input_dict2 = {
        "kernel_size": (5,),
        "stride": (3,),
        "padding": (2,),
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different kernel size, stride, and padding
    input3 = torch.randn(1, 5, 20).numpy()
    input_dict3 = {
        "kernel_size": (7,),
        "stride": (1,),
        "padding": (0,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4:  Negative values in input
    input4 = torch.randn(4, 2, 8).numpy() * -1
    input_dict4 = {
        "kernel_size": (2,),
        "stride": (2,),
        "padding": (0,),
        "ceil_mode": True,
        "count_include_pad": False,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger input
    input5 = torch.randn(3, 4, 30).numpy()
    input_dict5 = {
        "kernel_size": (4,),
        "stride": (3,),
        "padding": (1,),
        "ceil_mode": False,
        "count_include_pad": True,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AvgPool1d_2"] = torch_nn_AvgPool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AvgPool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool1d_2'.")

check_valid('torch.nn.AvgPool1d', generated_inputs['torch.nn.AvgPool1d_2'], lib="torch")
