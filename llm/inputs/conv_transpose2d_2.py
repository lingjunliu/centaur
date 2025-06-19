
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv_transpose2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1, 3, 10, 10).numpy()
    weight_tensor = torch.randn(3, 3, 3, 3).numpy()
    bias_tensor = torch.randn(3).numpy()
    stride = (1, 1)
    padding = (0, 0)
    output_padding = (0, 0)
    groups = 1
    dilation = (1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(2, 4, 12, 12).numpy()
    weight_tensor = torch.randn(4, 4, 5, 5).numpy()
    bias_tensor = torch.randn(4).numpy()
    stride = (2, 2)
    padding = (1, 1)
    output_padding = (1, 1)
    groups = 1
    dilation = (1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(1, 2, 8, 8).numpy()
    weight_tensor = torch.randn(2, 2, 2, 2).numpy()
    bias_tensor = torch.randn(2).numpy()
    stride = (3, 3)
    padding = (0, 0)
    output_padding = (0, 0)
    groups = 1
    dilation = (1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(4, 1, 16, 16).numpy()
    weight_tensor = torch.randn(1, 1, 7, 7).numpy()
    bias_tensor = torch.randn(1).numpy()
    stride = (1, 1)
    padding = (3, 3)
    output_padding = (0, 0)
    groups = 1
    dilation = (1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = torch.randn(1, 1, 5, 5).numpy()
    weight_tensor = torch.randn(1, 1, 2, 2).numpy()
    bias_tensor = torch.randn(1).numpy()
    stride = (2, 2)
    padding = (1, 1)
    output_padding = (1, 1)
    groups = 1
    dilation = (1, 1)

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.conv_transpose2d_2"] = conv_transpose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv_transpose2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv_transpose2d_2'.")

check_valid('torch.nn.functional.conv_transpose2d', generated_inputs['torch.nn.functional.conv_transpose2d_2'], lib="torch", suffix=2)
