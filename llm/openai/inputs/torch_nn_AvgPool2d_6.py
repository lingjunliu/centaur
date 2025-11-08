
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avgpool2d_inputs_6():
    list_of_inputs = []

    input_arr = torch.randn(2, 3, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 4, 50, 32, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 6,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 10, 5, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 4, 7, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 3,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = (torch.randn(4, 1, 9, 9, dtype=torch.float32) * 2.0).numpy()
    input_dict = {
        "kernel_size": (5, 5),
        "stride": (3, 3),
        "padding": 2,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 25,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.zeros(1, 1, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 2,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(5, 10, 11, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (4, 3),
        "stride": (4, 3),
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 12,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 13, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 5),
        "stride": (2, 3),
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 5,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 3, 20, 20, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (4, 4),
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 7, 15, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (7, 3),
        "stride": (5, 2),
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 10,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 16, 32, 17, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 4),
        "stride": (4, 2),
        "padding": 2,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 8,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_6"] = avgpool2d_inputs_6()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_6'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_6'], lib="torch", suffix=6)
