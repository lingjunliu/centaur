
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool3d_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = torch.randn(2, 3, 10, 8, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = torch.randn(1, 2, 7, 7, 7, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (4D input)
    input = torch.randn(3, 9, 9, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 3,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (dilation > 1)
    input = torch.randn(1, 4, 12, 10, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": 2,
        "padding": 1,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (ceil_mode True)
    input = torch.randn(1, 1, 5, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (return_indices True)
    input = torch.randn(2, 2, 6, 6, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 3,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (adjusted: padding compatible with kernel)
    input = torch.randn(1, 1, 4, 3, 2, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (5, 5, 5),
        "stride": 1,
        "padding": 2,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (dilation 3)
    input = torch.randn(1, 3, 9, 10, 11, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": 1,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (4D input, larger kernel/stride)
    input = torch.randn(4, 10, 12, 14, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": (4, 4, 4),
        "stride": 4,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (unit kernel)
    input = torch.randn(3, 3, 3, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1, 1),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (float tensor)
    input = torch.arange(2 * 3 * 12 * 8 * 10, dtype=torch.float32).reshape(2, 3, 12, 8, 10).numpy()
    input_dict = {
        "kernel_size": (2, 3, 4),
        "stride": 5,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_9"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_9' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_9'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_9'], lib="torch", suffix=9)
