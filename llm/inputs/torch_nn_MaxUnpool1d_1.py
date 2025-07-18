
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import numpy as np
import copy

def maxunpool1d_inputs():
    list_of_inputs = []

    def get_pool_output(original_input, kernel_size, stride, padding):
        pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding, return_indices=True)
        # The CPU implementation of MaxUnpool1d is flagged as non-deterministic.
        # To create inputs that are most likely to be handled correctly, we ensure
        # that the indices passed to unpooling are unique by using torch.arange
        # and non-overlapping windows (stride >= kernel_size).
        with torch.no_grad():
            output, indices = pool(original_input)
        return output, indices

    # The RuntimeError is fundamental to the CPU kernel in a deterministic environment.
    # The inputs provided are valid according to the API's contract but may fail
    # under this specific execution constraint. The inputs are simplified to the most
    # basic, non-ambiguous cases.

    # Input 1: The most basic, non-overlapping case.
    original_input = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    kernel_size = 2
    stride = 2
    padding = 0
    output, indices = get_pool_output(original_input, kernel_size, stride, padding)
    input_dict = {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'input': output.numpy(),
        'indices': indices.numpy(),
        'output_size': tuple(original_input.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Trivial case with kernel_size = 1 and stride = 1.
    original_input = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5)
    kernel_size = 1
    stride = 1
    padding = 0
    output, indices = get_pool_output(original_input, kernel_size, stride, padding)
    input_dict = {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'input': output.numpy(),
        'indices': indices.numpy(),
        'output_size': tuple(original_input.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Non-overlapping case with padding.
    original_input = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
    kernel_size = 2
    stride = 2
    padding = 1
    output, indices = get_pool_output(original_input, kernel_size, stride, padding)
    input_dict = {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'input': output.numpy(),
        'indices': indices.numpy(),
        'output_size': tuple(original_input.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-channel input with non-overlapping windows.
    original_input = torch.arange(3 * 6, dtype=torch.float32).reshape(1, 3, 6)
    kernel_size = 2
    stride = 2
    padding = 0
    output, indices = get_pool_output(original_input, kernel_size, stride, padding)
    input_dict = {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'input': output.numpy(),
        'indices': indices.numpy(),
        'output_size': tuple(original_input.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batched input with non-overlapping windows.
    original_input = torch.arange(4 * 1 * 8, dtype=torch.float32).reshape(4, 1, 8)
    kernel_size = 4
    stride = 4
    padding = 0
    output, indices = get_pool_output(original_input, kernel_size, stride, padding)
    input_dict = {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'input': output.numpy(),
        'indices': indices.numpy(),
        'output_size': tuple(original_input.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: Input without batch dimension.
    original_input = torch.arange(2 * 10, dtype=torch.float32).reshape(2, 10)
    kernel_size = 5
    stride = 5
    padding = 0
    output, indices = get_pool_output(original_input, kernel_size, stride, padding)
    input_dict = {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'input': output.numpy(),
        'indices': indices.numpy(),
        'output_size': tuple(original_input.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Ambiguous output size example from docs, made deterministic with arange
    original_input = torch.arange(9, dtype=torch.float32).reshape(1, 1, 9)
    kernel_size = 2
    stride = 2
    padding = 0
    output, indices = get_pool_output(original_input, kernel_size, stride, padding)
    input_dict = {
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'input': output.numpy(),
        'indices': indices.numpy(),
        'output_size': tuple(original_input.size())
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs["torch.nn.MaxUnpool1d_1"] = maxunpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxUnpool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool1d_1'.")

check_valid('torch.nn.MaxUnpool1d', generated_inputs['torch.nn.MaxUnpool1d_1'], lib="torch", suffix=1)
