
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def avgpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(2, 3, 10, 10, 10).numpy()
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
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
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 1, 5, 5, 5).numpy()
    kernel_size = (2, 2, 2)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    ceil_mode = True
    count_include_pad = False
    divisor_override = 2
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(4, 2, 8, 8, 8).numpy()
    kernel_size = (4, 4, 4)
    stride = (3, 3, 3)
    padding = (1, 1, 1)
    ceil_mode = False
    count_include_pad = True
    divisor_override = 5
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(1, 1, 7, 7, 7).numpy()
    kernel_size = (1, 1, 1)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    ceil_mode = True
    count_include_pad = False
    divisor_override = None
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(3, 4, 12, 12, 12).numpy()
    kernel_size = (5, 5, 5)
    stride = (4, 4, 4)
    padding = (2, 2, 2)
    ceil_mode = False
    count_include_pad = True
    divisor_override = 1
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input_tensor = torch.randn(1, 1, 3, 3, 3).numpy()
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
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
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(2, 2, 6, 6, 6).numpy()
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    ceil_mode = False
    count_include_pad = False
    divisor_override = None
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(1, 3, 9, 9, 9).numpy()
    kernel_size = (3, 2, 1)
    stride = (1, 2, 3)
    padding = (0, 1, 0)  # Reduced padding to avoid error
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
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input_tensor = torch.randn(1, 1, 4, 4, 4).numpy()
    kernel_size = (4, 4, 4)
    stride = (4, 4, 4)
    padding = (0, 0, 0)
    ceil_mode = False
    count_include_pad = False
    divisor_override = 1
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = torch.randn(5, 1, 15, 15, 15).numpy()
    kernel_size = (2, 2, 2)
    stride = (3, 3, 3)
    padding = (0, 0, 0)  # Reduced padding to avoid error
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
        "input": input_tensor,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool3d_2"] = avgpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.AvgPool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool3d_2'.")

check_valid('torch.nn.AvgPool3d', generated_inputs['torch.nn.AvgPool3d_2'], lib="torch", suffix=2)
