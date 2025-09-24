
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def generate_avgpool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 3, 10, 10).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(2, 3, 20, 20).numpy()
    input_dict = {
        "kernel_size": (4, 4),
        "stride": 2,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 - Modified to fix the padding issue
    input_tensor = torch.randn(2, 1, 15, 15).numpy()
    input_dict = {
        "kernel_size": (5, 5),
        "stride": 3,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 10,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(1, 1, 7, 7).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": 1,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(1, 3, 12, 12).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(2, 3, 25, 25).numpy()
    input_dict = {
        "kernel_size": (5, 3),
        "stride": 3,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(2, 1, 18, 18).numpy()
    input_dict = {
        "kernel_size": (4, 4),
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 8,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = torch.randn(1, 1, 6, 6).numpy()
    input_dict = {
        "kernel_size": (1, 1),
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = torch.randn(1, 3, 9, 9).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": 3,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 5,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AvgPool2d_5"] = generate_avgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AvgPool2d_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_5'.")

check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_5'], lib="torch", suffix=5)
