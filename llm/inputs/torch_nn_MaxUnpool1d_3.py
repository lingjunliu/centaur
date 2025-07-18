
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import torch.nn as nn
import numpy as np
import copy

def maxunpool1d_inputs():
    list_of_inputs = []

    # The recurrent error "max_unpooling2d_forward_out does not have a deterministic implementation"
    # is due to the execution environment having `torch.use_deterministic_algorithms(True)`.
    # The CPU implementation of MaxUnpool1d in PyTorch is not flagged as deterministic, causing this
    # error regardless of the input values. No input can bypass this internal library check.
    # The only recourse is to provide the most simple, canonical example from the PyTorch documentation
    # to demonstrate that the input is valid and the issue lies with the environment's constraints.

    # Case 1: The most basic example from the official documentation.
    kernel_size = 2
    stride = 2
    padding = 0
    pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding, return_indices=True)
    original_input = torch.tensor([[[1., 2., 3., 4., 5., 6., 7., 8.]]], dtype=torch.float32)
    output, indices = pool(original_input)
    
    input_dict = {
        'kernel_size': kernel_size,
        'stride': (stride,),
        'padding': padding,
        'input': output.numpy(),
        'indices': indices.numpy(),
        'output_size': tuple(original_input.shape)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxUnpool1d_3"] = maxunpool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxUnpool1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxUnpool1d_3'.")

check_valid('torch.nn.MaxUnpool1d', generated_inputs['torch.nn.MaxUnpool1d_3'], lib="torch", suffix=3)
