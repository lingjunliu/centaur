
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def unfold_inputs():
    list_of_inputs = []

    # Input 1: Basic case with common parameters
    input1 = torch.randn(2, 3, 10, 12).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": (3, 4),
        "dilation": (1, 1),
        "padding": (0, 0),
        "stride": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different kernel size, stride, and padding
    input2 = torch.randn(1, 5, 8, 8).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": (2, 2),
        "dilation": (1, 1),
        "padding": (1, 1),
        "stride": (2, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Dilation > 1
    input3 = torch.randn(1, 2, 15, 15).numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": (3, 3),
        "dilation": (2, 2),
        "padding": (0, 0),
        "stride": (1, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Asymmetric kernel, padding, stride
    input4 = torch.randn(4, 4, 7, 9).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": (2, 3),
        "dilation": (1, 1),
        "padding": (1, 0),
        "stride": (1, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Large input size
    input5 = torch.randn(1, 1, 32, 32).numpy()
    input_dict5 = {
        "input": input5,
        "kernel_size": (5, 5),
        "dilation": (1, 1),
        "padding": (2, 2),
        "stride": (3, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs["torch.nn.Unfold_2"] = unfold_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.Unfold_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Unfold_2'.")

check_valid('torch.nn.Unfold', generated_inputs['torch.nn.Unfold_2'], lib="torch")
