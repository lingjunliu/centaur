
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def maxpool3d_inputs():
    list_of_inputs = []

    inp = torch.randn(1, 1, 4, 4, 4).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 5, 6, 7).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": (2, 1, 2),
        "padding": (1, 0, 1),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 4, 8, 9, 10).numpy()
    input_dict = {
        "kernel_size": (2, 2, 3),
        "stride": (1, 2, 1),
        "padding": (1, 1, 1),
        "dilation": (2, 2, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 6, 7, 8).numpy()
    input_dict = {
        "kernel_size": (1, 3, 1),
        "stride": (1, 2, 1),
        "padding": (0, 1, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 5, 10, 9, 8).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(4, 2, 11, 5, 13).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (3, 2, 4),
        "padding": (0, 1, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 3, 3, 4).numpy()
    input_dict = {
        "kernel_size": (4, 1, 5),
        "stride": (1, 1, 1),
        "padding": (2, 0, 2),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = (torch.randn(2, 3, 2, 2, 2) * -2.0).numpy()
    input_dict = {
        "kernel_size": (1, 1, 1),
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 2, 6, 10, 7).numpy()
    input_dict = {
        "kernel_size": (2, 3, 4),
        "stride": (2, 3, 1),
        "padding": (1, 1, 2),
        "dilation": (1, 2, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 5, 5, 5).numpy()
    input_dict = {
        "kernel_size": (5, 5, 5),
        "stride": (3, 3, 3),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 2, 4, 3, 3).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": (2, 1, 3),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 7, 8, 9).numpy()
    input_dict = {
        "kernel_size": (1, 2, 2),
        "stride": (1, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_16"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_16' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_16'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_16'], lib="torch", suffix=16)
