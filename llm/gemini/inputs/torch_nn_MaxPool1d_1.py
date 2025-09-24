
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def maxpool1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1, 1, 10).numpy()
    kernel_size = 3
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(2, 3, 20).numpy()
    kernel_size = 4
    stride = 1
    padding = 1
    dilation = 2
    return_indices = True
    ceil_mode = True

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(5, 2, 15).numpy()
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = True

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(1, 1, 5).numpy()
    kernel_size = 5
    stride = 1
    padding = 2
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(3, 4, 12).numpy()
    kernel_size = 3
    stride = 3
    padding = 1
    dilation = 1
    return_indices = True
    ceil_mode = False

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = torch.randn(2, 2, 8).numpy()
    kernel_size = 2
    stride = 1
    padding = 1
    dilation = 2
    return_indices = False
    ceil_mode = True

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(4, 1, 10).numpy()
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True
    ceil_mode = True

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = torch.randn(1, 5, 25).numpy()
    kernel_size = 5
    stride = 3
    padding = 2
    dilation = 2
    return_indices = False
    ceil_mode = False

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = torch.randn(6, 3, 18).numpy()
    kernel_size = 4
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True
    ceil_mode = True

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.randn(1, 1, 7).numpy()
    kernel_size = 7
    stride = 1
    padding = 3
    dilation = 1
    return_indices = False
    ceil_mode = False

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxPool1d_1"] = maxpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxPool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool1d_1'.")

check_valid('torch.nn.MaxPool1d', generated_inputs['torch.nn.MaxPool1d_1'], lib="torch", suffix=1)
