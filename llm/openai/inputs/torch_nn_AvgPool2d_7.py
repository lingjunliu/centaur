
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avgpool2d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(2, 3, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": 2,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 1, 7, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": (1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 10, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 3),
        "stride": 2,
        "padding": (1, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 12,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 8, 32, 16, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (5, 2),
        "stride": 3,
        "padding": (2, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 10,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones(1, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": 1,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 5, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": 2,
        "padding": (1, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 6,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 9, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": 4,
        "padding": (1, 1),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(5, 13, 13, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (6, 6),
        "stride": 5,
        "padding": (2, 2),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 36,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 4, 6, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1),
        "stride": 1,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 1, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 4, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 3),
        "stride": 2,
        "padding": (0, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 6,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 5, 10, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 1),
        "stride": 1,
        "padding": (2, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 1, 1, 1, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1),
        "stride": 1,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_7"] = avgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_7'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_7'], lib="torch", suffix=7)
