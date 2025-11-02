
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(2, 3, 5, 6, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": (1, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(1, 2, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(1, 1, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.randn(1, 1, 6, 6, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (4D input: C, D, H, W)
    input_arr = torch.randn(3, 10, 12, 14, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 5, 6),
        "stride": (4, 4, 4),
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (float64)
    input_arr = torch.randn(2, 4, 7, 8, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (2, 3, 2),
        "stride": (2, 3, 2),
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (dilation > 1 with padding adjusted to be valid)
    input_arr = torch.randn(1, 2, 6, 6, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": 1,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (negative values)
    input_arr = torch.full((1, 1, 3, 3, 3), -5.0, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 1, 1),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (small input, padding adjusted to be valid)
    input_arr = torch.randn(1, 1, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1, 1),
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (larger multi-channel with dilation)
    input_arr = torch.randn(4, 8, 15, 13, 11, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (1, 2, 2),
        "padding": 1,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (4D input with ceil_mode)
    input_arr = torch.randn(5, 9, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 4, 5),
        "stride": (3, 4, 5),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_13"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_13' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_13'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_13'], lib="torch", suffix=13)
