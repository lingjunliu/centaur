
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def convtranspose3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(2, 3, 10, 10, 10).float().numpy()
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
        "dtype": np.dtype('float32'),
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 16, 5, 5, 5).float().numpy()
    input_dict = {
        "in_channels": 16,
        "out_channels": 32,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "bias": False,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.dtype('float32'),
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(4, 8, 7, 7, 7).float().numpy()
    input_dict = {
        "in_channels": 8,
        "out_channels": 4,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.dtype('float32'),
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(1, 3, 10, 10, 10).float().numpy()
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
        "dtype": np.dtype('float32'),
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(2, 4, 6, 6, 6).float().numpy()
    input_dict = {
        "in_channels": 4,
        "out_channels": 8,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "output_padding": 1,
        "groups": 2,
        "bias": True,
        "dilation": 2,
        "padding_mode": "zeros",
        "dtype": np.dtype('float32'),
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    input_tensor = torch.randn(1, 6, 8, 8, 8).float().numpy()
    input_dict = {
        "in_channels": 6,
        "out_channels": 12,
        "kernel_size": 5,
        "stride": 3,
        "padding": 2,
        "output_padding": 1,
        "groups": 3,
        "bias": True,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.dtype('float32'),
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = torch.randn(3, 7, 9, 9, 9).float().numpy()
    input_dict = {
        "in_channels": 7,
        "out_channels": 14,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "output_padding": 0,
        "groups": 1,
        "bias": False,
        "dilation": 2,
        "padding_mode": "zeros",
        "dtype": np.dtype('float32'),
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8
    input_tensor = torch.randn(1, 2, 4, 4, 4).float().numpy()
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
        "dtype": np.dtype('float32'),
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = torch.randn(2, 5, 3, 3, 3).float().numpy()
    input_dict = {
        "in_channels": 5,
        "out_channels": 10,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "bias": False,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.dtype('float32'),
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = torch.randn(1, 1, 2, 2, 2).float().numpy()
    input_dict = {
        "in_channels": 1,
        "out_channels": 1,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "groups": 1,
        "bias": True,
        "dilation": 1,
        "padding_mode": "zeros",
        "dtype": np.dtype('float32'),
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.ConvTranspose3d"] = convtranspose3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.ConvTranspose3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConvTranspose3d'.")

check_valid('torch.nn.ConvTranspose3d', generated_inputs['torch.nn.ConvTranspose3d'], lib="torch", suffix=0)
