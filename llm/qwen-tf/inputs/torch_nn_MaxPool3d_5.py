
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def maxpool3d_inputs():
    list_of_inputs = []

    input = torch.randn(1, 1, 8, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 10, 12, 14, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 1, 2),
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2, 2),
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 10, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 1, 1),
        "padding": 1,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 7, 7, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (3, 3, 3),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 9, 9, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (3, 3, 3),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 5, 20, 15, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": (4, 3, 2),
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 3, 18, 18, 18, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": 1,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 4, 6, 7, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (1, 2, 3),
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 3, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1, 1),
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 10, 8, 10, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": (2, 5, 4),
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 11, 13, 17, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 4, 5),
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_5"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_5'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_5'], lib="torch", suffix=5)
