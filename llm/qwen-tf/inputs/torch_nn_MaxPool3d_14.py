
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool3d_inputs_14():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(2, 3, 8, 10, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2 (4D input)
    input_arr = torch.randn(1, 7, 9, 11, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (1, 2, 2),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(1, 2, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 1, 1),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": (2, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(3, 4, 9, 9, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 3, 4),
        "stride": (1, 2, 3),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (dilation > 1, padding adjusted to be valid)
    input_arr = torch.randn(1, 1, 10, 12, 14, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": 1,
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (ceil_mode=True)
    input_arr = torch.randn(2, 2, 7, 8, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": (3, 3, 3),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (padding > 0)
    input_arr = torch.randn(1, 3, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 1),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (4D input, non-cubic kernel)
    input_arr = torch.randn(2, 8, 6, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 4, 3),
        "stride": (2, 3, 2),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (4D input, ceil_mode=True, padding > 0)
    input_arr = torch.randn(1, 6, 6, 6, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (2, 2, 2),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (mixed dilation)
    input_arr = torch.randn(2, 5, 5, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 2, 3),
        "stride": (1, 2, 1),
        "padding": 0,
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (dilation with stride=1)
    input_arr = torch.randn(1, 1, 15, 15, 15, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": (3, 1, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (minimal spatial dims with kernel=1)
    input_arr = torch.randn(4, 1, 1, 1, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1, 1),
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_14"] = maxpool3d_inputs_14()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_14' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_14'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_14'], lib="torch", suffix=14)
