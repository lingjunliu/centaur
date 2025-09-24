
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(1, 1, 4, 4, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2 (4D input)
    input_arr = torch.randn(3, 5, 6, 7).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2, 2),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3 (dilation > 1, float64)
    input_arr = torch.randn(2, 2, 5, 5, 5, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (non-uniform stride, return_indices True)
    input_arr = torch.randn(4, 8, 9, 10, 11).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (3, 2, 4),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (ceil_mode True)
    input_arr = torch.randn(1, 1, 5, 5, 5).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (valid padding)
    input_arr = torch.randn(1, 1, 1, 3, 3).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1, 1),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (return_indices True, mixed stride)
    input_arr = torch.randn(2, 3, 7, 9, 11).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 3, 4),
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (mixed dilation, ceil_mode True, adjusted padding)
    input_arr = torch.randn(3, 1, 6, 6, 6).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2, 2),
        "padding": 1,
        "dilation": (2, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (4D input, float64)
    input_arr = torch.randn(4, 5, 8, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (k=1 with dilation > 1)
    input_arr = torch.ones(5, 6, 7, 8, 9).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (1, 1, 1),
        "padding": 0,
        "dilation": (3, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_6"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_6'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_6'], lib="torch", suffix=6)
