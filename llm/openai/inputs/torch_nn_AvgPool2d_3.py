
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avgpool2d_inputs_3():
    list_of_inputs = []

    # Input 1
    input_arr = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2),
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(2, 3, 7, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 1),
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 (CHW)
    input_arr = torch.randn(3, 8, 8, dtype=torch.double).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 1),
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(4, 2, 10, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": (3, 4),
        "padding": 2,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 25,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.ones(1, 4, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (1, 1),
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.full((1, 1, 5, 5), -5.0, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2),
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.randn(1, 5, 9, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 3),
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (CHW)
    input_arr = torch.randn(8, 15, 17, dtype=torch.double).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": (4, 4),
        "padding": 2,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 16,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (single element)
    input_arr = torch.tensor([[[[3.0]]]], dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (1, 1),
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (CHW, non-square stride)
    input_arr = torch.randn(6, 11, 13, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": (3, 2),
        "padding": 2,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 25,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (larger padding within limit)
    input_arr = (torch.ones(2, 1, 8, 8, dtype=torch.float32) * -2.5).numpy()
    input_dict = {
        "kernel_size": 7,
        "stride": (3, 3),
        "padding": 3,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 49,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_3"] = avgpool2d_inputs_3()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_3'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_3'], lib="torch", suffix=3)
