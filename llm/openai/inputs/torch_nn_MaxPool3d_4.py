
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def maxpool3d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(2, 3, 10, 10, 10, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(1, 2, 7, 7, 7, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 3,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(1, 1, 5, 6, 7, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 3,
        "stride": 2,
        "padding": (1, 1, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(3, 2, 5, 5, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 2,
        "stride": 1,
        "padding": (0, 0, 0),
        "dilation": (2, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(2, 4, 9, 8, 7, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 3,
        "stride": 4,
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }))

    input_arr = torch.randn(3, 8, 8, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(1, 4, 5, 6, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 3,
        "stride": 1,
        "padding": (1, 0, 1),
        "dilation": (2, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = (-torch.rand(1, 5, 3, 4, 5, dtype=torch.float32)).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 1,
        "stride": 1,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(2, 2, 5, 5, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 2,
        "stride": 1,
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(1, 1, 15, 16, 17, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 5,
        "stride": 2,
        "padding": (2, 2, 2),
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(1, 1, 1, 1, 1, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 1,
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }))

    input_arr = torch.randn(4, 3, 12, 10, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "kernel_size": 4,
        "stride": 3,
        "padding": (2, 1, 0),
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_4"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_4'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_4'], lib="torch", suffix=4)
