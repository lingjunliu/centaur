
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv_transpose2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input_dict = {
        "in_channels": 16,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": torch.randn(1, 16, 5, 5).numpy().astype(np.float32),
        "output_size": (1, 32, 7, 7)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Stride > 1 (upsampling), float64
    input_dict = {
        "in_channels": 8,
        "out_channels": 16,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "output_padding": 0,
        "groups": 1,
        "bias": False,
        "dilation": 1,
        "padding_mode": 'zeros',
        "dtype": torch.float64,
        "input": torch.randn(4, 8, 10, 10).numpy().astype(np.float64),
        "output_size": (4, 16, 20, 20)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: `groups` > 1
    input_dict = {
        "in_channels": 16,
        "out_channels": 32,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "output_padding": 0,
        "groups": 8,
        "bias": True,
        "dilation": 1,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": torch.randn(1, 16, 12, 12).numpy().astype(np.float32),
        "output_size": (1, 32, 12, 12)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: `groups` == `in_channels` == `out_channels` (depthwise)
    input_dict = {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 5,
        "stride": 1,
        "padding": 2,
        "output_padding": 0,
        "groups": 16,
        "bias": True,
        "dilation": 1,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": torch.randn(1, 16, 8, 8).numpy().astype(np.float32),
        "output_size": (1, 16, 8, 8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: `dilation` > 1
    input_dict = {
        "in_channels": 3,
        "out_channels": 6,
        "kernel_size": 3,
        "stride": 1,
        "padding": 2,
        "output_padding": 0,
        "groups": 1,
        "bias": False,
        "dilation": 2,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": torch.randn(2, 3, 20, 20).numpy().astype(np.float32),
        "output_size": (2, 6, 20, 20)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: With `output_padding` to resolve shape ambiguity
    input_dict = {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": torch.randn(1, 16, 6, 6).numpy().astype(np.float32),
        "output_size": (1, 16, 12, 12)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Using `output_size` to specify an exact shape
    input_dict = {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": torch.randn(1, 16, 6, 6).numpy().astype(np.float32),
        "output_size": (1, 16, 12, 12)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: No batch dimension in input
    input_dict = {
        "in_channels": 4,
        "out_channels": 8,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "output_padding": 0,
        "groups": 2,
        "bias": False,
        "dilation": 1,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": torch.randn(4, 7, 7).numpy().astype(np.float32),
        "output_size": (8, 14, 14)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large kernel size
    input_dict = {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 7,
        "stride": 1,
        "padding": 3,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": torch.randn(1, 1, 32, 32).numpy().astype(np.float32),
        "output_size": (1, 1, 32, 32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Combination of dilation, groups, and output_padding
    input_dict = {
        "in_channels": 10,
        "out_channels": 20,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "groups": 10,
        "bias": False,
        "dilation": 2,
        "padding_mode": 'zeros',
        "dtype": torch.float64,
        "input": torch.randn(1, 10, 15, 15).numpy().astype(np.float64),
        "output_size": (1, 20, 32, 32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Large number of channels
    input_dict = {
        "in_channels": 256,
        "out_channels": 128,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "output_padding": 0,
        "groups": 4,
        "bias": True,
        "dilation": 1,
        "padding_mode": 'zeros',
        "dtype": torch.float32,
        "input": torch.randn(1, 256, 8, 8).numpy().astype(np.float32),
        "output_size": (1, 128, 16, 16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.ConvTranspose2d"] = conv_transpose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConvTranspose2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConvTranspose2d'.")

check_valid('torch.nn.ConvTranspose2d', generated_inputs['torch.nn.ConvTranspose2d'], lib="torch", suffix=0)
