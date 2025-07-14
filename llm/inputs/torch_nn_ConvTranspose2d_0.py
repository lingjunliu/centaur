
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def convtranspose2d_inputs():
    list_of_inputs = []

    # Input 1
    input = np.random.randn(1, 3, 10, 10).astype(np.float32)
    input_dict = {
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.float32,
        "input": input,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.random.randn(2, 4, 8, 8).astype(np.float64)
    input_dict = {
        "in_channels": 4,
        "out_channels": 8,
        "kernel_size": 5,
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "groups": 2,
        "bias": False,
        "dilation": 2,
        "padding_mode": "zeros",
        "dtype": np.float64,
        "input": input,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.random.randn(1, 16, 12, 12).astype(np.float32)
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
        "padding_mode": "zeros",
        "dtype": np.float32,
        "input": input,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input = np.random.randn(4, 3, 16, 16).astype(np.float32)
    input_dict = {
        "in_channels": 3,
        "out_channels": 7,
        "kernel_size": 4,
        "stride": 3,
        "padding": 1,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.float32,
        "input": input,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.random.randn(1, 2, 5, 5).astype(np.float32)
    input_dict = {
        "in_channels": 2,
        "out_channels": 4,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.float32,
        "input": input,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input = np.random.randn(1, 3, 10, 10).astype(np.float32)
    input_dict = {
        "in_channels": 3,
        "out_channels": 3,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "output_padding": 0,
        "groups": 3,
        "bias": False,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.float32,
        "input": input,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input = np.random.randn(2, 4, 8, 8).astype(np.float64)
    input_dict = {
        "in_channels": 4,
        "out_channels": 4,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "output_padding": 0,
        "groups": 4,
        "bias": True,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.float64,
        "input": input,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input = np.random.randn(1, 8, 6, 6).astype(np.float32)
    input_dict = {
        "in_channels": 8,
        "out_channels": 4,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.float32,
        "input": input,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input = np.random.randn(4, 2, 7, 7).astype(np.float32)
    input_dict = {
        "in_channels": 2,
        "out_channels": 4,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "output_padding": 0,
        "groups": 1,
        "bias": False,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.float32,
        "input": input,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input = np.random.randn(1, 3, 10, 10).astype(np.float32)
    input_dict = {
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "output_padding": 1,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.float32,
        "input": input,
        "output_size": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ConvTranspose2d"] = convtranspose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConvTranspose2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConvTranspose2d'.")

check_valid('torch.nn.ConvTranspose2d', generated_inputs['torch.nn.ConvTranspose2d'], lib="torch", suffix=0)
