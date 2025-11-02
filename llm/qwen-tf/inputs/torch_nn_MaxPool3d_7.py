
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool3d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(1, 1, 8, 8, 8).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 10, 12, 14).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 4, 20, 15, 15).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2, 2),
        "padding": (1, 1, 1),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 7, 7, 7).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": (3, 3, 3),
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 6, 16, 17, 9).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": (2, 3, 2),
        "padding": (2, 1, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(5, 3, 3, 3).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1, 1),
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 2, 25, 18, 33).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (3, 2, 4),
        "padding": (0, 1, 1),
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(8, 1, 9, 20, 11, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2, 1),
        "padding": (0, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 7, 9, 11).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(5, 7, 10, 13, 20).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 3, 4),
        "padding": (1, 0, 1),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 2, 60, 33, 25, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 7,
        "stride": (6, 5, 4),
        "padding": (3, 2, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 4, 4, 4).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_7"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_7'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_7'], lib="torch", suffix=7)
