
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def maxpool3d_inputs():
    list_of_inputs = []

    inp = torch.randn(2, 3, 8, 10, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 7, 9, 11, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 6, 7, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 8, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 4,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 4, 6, 6, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(5, 2, 5, 6, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": 1,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 4, 9, 9, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 2,
        "padding": 2,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 1, 9, 9, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(7, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 2, 3, 2, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 11, 7, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 5,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_1"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_1'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_1'], lib="torch", suffix=1)
