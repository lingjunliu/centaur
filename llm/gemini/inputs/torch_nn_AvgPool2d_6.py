
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def avgpool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(2, 3, 20, 20).numpy()
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = 0
    ceil_mode = False
    count_include_pad = True
    divisor_override = None

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 1, 25, 25).numpy()
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 1
    ceil_mode = False
    count_include_pad = False
    divisor_override = 5

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(4, 5, 30, 30).numpy()
    kernel_size = (5, 5)
    stride = (3, 3)
    padding = 2
    ceil_mode = True
    count_include_pad = True
    divisor_override = None

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(1, 1, 10, 10).numpy()
    kernel_size = (2, 2)
    stride = (2, 2)
    padding = 0
    ceil_mode = True
    count_include_pad = False
    divisor_override = 1

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(3, 2, 15, 15).numpy()
    kernel_size = (4, 4)
    stride = (1, 1)
    padding = 1
    ceil_mode = False
    count_include_pad = True
    divisor_override = 10

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(1, 1, 7, 7).numpy()
    kernel_size = (1, 1)
    stride = (1, 1)
    padding = 0
    ceil_mode = False
    count_include_pad = True
    divisor_override = None

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(2, 3, 12, 12).numpy()
    kernel_size = (3, 2)
    stride = (2, 1)
    padding = 1
    ceil_mode = True
    count_include_pad = False
    divisor_override = 3

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(1, 1, 8, 8).numpy()
    kernel_size = (4, 3)
    stride = (1, 2)
    padding = 0
    ceil_mode = False
    count_include_pad = True
    divisor_override = None

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = torch.randn(4, 2, 16, 16).numpy()
    kernel_size = (2, 2)
    stride = (3, 1)
    padding = 0
    ceil_mode = True
    count_include_pad = False
    divisor_override = 7

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.randn(1, 1, 9, 9).numpy()
    kernel_size = (5, 2)
    stride = (2, 3)
    padding = 1
    ceil_mode = False
    count_include_pad = True
    divisor_override = None

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_6"] = avgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AvgPool2d_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_6'.")

check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_6'], lib="torch", suffix=6)
