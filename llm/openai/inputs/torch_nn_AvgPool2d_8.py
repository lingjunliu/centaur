
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def avgpool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(2, 3, 8, 8).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 2,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(3, 10, 7).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2),
        "padding": (1, 1),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 3,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(4, 16, 20, 15).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": (4, 3),
        "padding": (2, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 5,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.randn(1, 3, 9, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.randn(3, 5, 3).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 1),
        "padding": (0, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 3,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.randn(5, 7, 7, 7).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (2, 3),
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.randn(2, 4, 11, 10, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": (3, 2),
        "padding": (2, 1),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.randn(6, 1, 6, 6, dtype=torch.float16).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2),
        "padding": (1, 1),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 5,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.randn(1, 6, 9).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 3),
        "padding": (0, 0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 2,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = torch.randn(8, 2, 13, 17).numpy()
    input_dict = {
        "kernel_size": 7,
        "stride": (5, 4),
        "padding": (3, 2),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 7,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.randn(1, 1, 2, 10).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 3),
        "padding": (0, 1),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 2,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_8"] = avgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_8'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_8'], lib="torch", suffix=8)
