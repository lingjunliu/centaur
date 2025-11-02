
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def maxpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(2, 4, 10, 12, 14).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(20, 16, 50, 44, 31).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": (2, 1, 2),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(1, 1, 3, 3, 3).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (fixed padding to satisfy pad <= floor(kernel/2))
    input_arr = torch.randn(1, 2, 8, 9, 10).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.randn(3, 5, 9, 9, 9).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (4D input without batch)
    input_arr = torch.randn(4, 12, 13, 14).numpy()
    input_dict = {
        "kernel_size": (2, 3, 4),
        "stride": (2, 3, 4),
        "padding": (0, 1, 2),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (use float32 to ensure broad CPU support)
    input_arr = torch.randn(2, 3, 16, 16, 16, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1, 1),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.randn(1, 2, 7, 8, 9).numpy()
    input_dict = {
        "kernel_size": (4, 4, 4),
        "stride": (3, 3, 3),
        "padding": (2, 1, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (zeros)
    input_arr = torch.zeros(1, 1, 5, 5, 5).numpy()
    input_dict = {
        "kernel_size": (5, 5, 5),
        "stride": (5, 5, 5),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (dilation 3)
    input_arr = torch.randn(2, 2, 10, 10, 10).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (4D input, negative-heavy)
    input_arr = (torch.randn(1, 7, 7, 7) * 10.0).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 2, 3),
        "padding": (0, 0, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (prime sizes with dilation 2)
    input_arr = torch.randn(4, 11, 17, 19).numpy()
    input_dict = {
        "kernel_size": (1, 4, 2),
        "stride": (2, 1, 2),
        "padding": (0, 1, 0),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_15"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_15' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_15'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_15'], lib="torch", suffix=15)
