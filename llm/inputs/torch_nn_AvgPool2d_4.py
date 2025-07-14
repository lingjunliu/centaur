
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def generate_avgpool2d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = torch.randn(20, 16, 50, 32).numpy()
    input_dict1 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1,
        "input": input1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = torch.randn(1, 1, 10, 10).numpy()
    input_dict2 = {
        "kernel_size": 2,
        "stride": 2,
        "padding": (1, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 2,
        "input": input2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = torch.randn(5, 3, 25, 25).numpy()
    input_dict3 = {
        "kernel_size": 5,
        "stride": 1,
        "padding": (2, 2),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 5,
        "input": input3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = torch.randn(10, 8, 40, 60).numpy()
    input_dict4 = {
        "kernel_size": 4,
        "stride": 4,
        "padding": (1, 0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": None,
        "input": input4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

     # Input 5
    input5 = torch.randn(2, 4, 12, 12).numpy()
    input_dict5 = {
        "kernel_size": 3,
        "stride": 2,
        "padding": (0, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = torch.randn(3, 2, 8, 8).numpy()
    input_dict6 = {
        "kernel_size": 2,
        "stride": 1,
        "padding": (1, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": None,
        "input": input6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = torch.randn(4, 5, 16, 16).numpy()
    input_dict7 = {
        "kernel_size": 4,
        "stride": 2,
        "padding": (2, 2),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 16,
        "input": input7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = torch.randn(1, 1, 20, 20).numpy()
    input_dict8 = {
        "kernel_size": 5,
        "stride": 5,
        "padding": (0, 0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": None,
        "input": input8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Input 9
    input9 = torch.randn(1, 1, 3, 3).numpy()
    input_dict9 = {
        "kernel_size": 1,
        "stride": 1,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = torch.randn(2, 2, 7, 7).numpy()
    input_dict10 = {
        "kernel_size": 2,
        "stride": 1,
        "padding": (0, 0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 4,
        "input": input10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.AvgPool2d_4"] = generate_avgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AvgPool2d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_4'.")

check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_4'], lib="torch", suffix=4)
