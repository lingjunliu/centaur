
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def max_unpool3d_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with N, C, Din, Hin, Win
    input_shape = (1, 1, 2, 2, 2)
    kernel_size = 1
    stride = 1
    padding = 0

    pool = torch.nn.MaxPool3d(kernel_size, stride=stride, padding=padding, return_indices=True)
    input_tensor = torch.randn(input_shape)
    output, indices = pool(input_tensor)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Simplest
    input_shape = (1, 1, 1, 1, 1)
    kernel_size = 1
    stride = 1
    padding = 0

    pool = torch.nn.MaxPool3d(kernel_size, stride=stride, padding=padding, return_indices=True)
    input_tensor = torch.randn(input_shape)
    output, indices = pool(input_tensor)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Slightly larger, still small
    input_shape = (1, 1, 3, 3, 3)
    kernel_size = 1
    stride = 1
    padding = 0

    pool = torch.nn.MaxPool3d(kernel_size, stride=stride, padding=padding, return_indices=True)
    input_tensor = torch.randn(input_shape)
    output, indices = pool(input_tensor)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Different kernel size, small
    input_shape = (1, 1, 3, 3, 3)
    kernel_size = 2
    stride = 1
    padding = 0

    pool = torch.nn.MaxPool3d(kernel_size, stride=stride, padding=padding, return_indices=True)
    input_tensor = torch.randn(input_shape)
    output, indices = pool(input_tensor)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 5: Adding a channel dimension
    input_shape = (1, 2, 3, 3, 3)
    kernel_size = 1
    stride = 1
    padding = 0

    pool = torch.nn.MaxPool3d(kernel_size, stride=stride, padding=padding, return_indices=True)
    input_tensor = torch.randn(input_shape)
    output, indices = pool(input_tensor)

    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input": output.numpy(),
        "indices": indices.numpy(),
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.MaxUnpool3d_1"] = max_unpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.MaxUnpool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool3d_1'.")

check_valid('torch.nn.MaxUnpool3d', generated_inputs['torch.nn.MaxUnpool3d_1'], lib="torch")
