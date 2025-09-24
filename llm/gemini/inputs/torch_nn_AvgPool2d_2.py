
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
        "stride": (2, 2),
        "padding": (0, 0),
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
        "stride": (1, 1),
        "padding": (1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(2, 3, 20, 20).numpy()
    input_dict = {
        "kernel_size": (4, 4),
        "stride": (2, 2),
        "padding": (2, 2),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(1, 1, 7, 7).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 5,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(1, 3, 15, 15).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": (1, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 2,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Test with small input size
    input_tensor = torch.randn(1, 1, 2, 2).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Test with different padding
    input_tensor = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (1, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Test with stride greater than kernel size
    input_tensor = torch.randn(1, 1, 10, 10).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (3, 3),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Non-square input
    input_tensor = torch.randn(1, 1, 5, 7).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large divisor_override
    input_tensor = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 10,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AvgPool2d_2"] = generate_avgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AvgPool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_2'.")

check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_2'], lib="torch", suffix=2)
