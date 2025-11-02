
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(2, 3, 10, 12, 14).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (3, 3, 3),
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 2
    input_arr = torch.randn(3, 8, 8, 8).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 3
    input_arr = torch.randn(1, 1, 5, 7, 9).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (3, 2, 2),
        "stride": 1,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }))

    # Input 4
    input_arr = torch.randn(4, 2, 6, 6, 6).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (3, 3, 3),
        "stride": 1,
        "padding": 1,
        "dilation": (2, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 5
    input_arr = torch.randn(2, 1, 5, 5, 5).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (4, 4, 4),
        "stride": 3,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }))

    # Input 6
    input_arr = torch.randn(1, 3, 7, 8, 10).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": (1, 2, 3),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 7
    input_arr = torch.randn(2, 1, 20, 3, 7, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (5, 1, 3),
        "stride": 1,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 8
    input_arr = torch.randn(4, 9, 9, 9).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (1, 3, 3),
        "stride": 2,
        "padding": 0,
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }))

    # Input 9
    input_arr = torch.randn(8, 4, 15, 17, 19).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (2, 3, 2),
        "stride": 3,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }))

    # Input 10
    input_arr = torch.randn(1, 5, 16, 16, 16).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (4, 2, 2),
        "stride": 4,
        "padding": 0,
        "dilation": (2, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 11
    input_arr = torch.randn(3, 2, 6, 7, 5).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (2, 3, 2),
        "stride": 2,
        "padding": 1,
        "dilation": (3, 2, 2),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    # Input 12
    input_arr = torch.randn(2, 12, 11, 10).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": (3, 4, 5),
        "stride": 2,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_10"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_10' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_10'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_10'], lib="torch", suffix=10)
