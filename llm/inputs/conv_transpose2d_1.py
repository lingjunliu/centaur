
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def conv_transpose2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1, 1, 5, 5).numpy()
    weight_tensor = torch.randn(1, 1, 3, 3).numpy()
    bias_tensor = torch.randn(1).numpy()
    stride_val = 1
    padding_val = 0
    output_padding_val = 0
    groups_val = 1
    dilation_val = 1
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 3, 10, 10).numpy()
    weight_tensor = torch.randn(3, 1, 5, 5).numpy()
    bias_tensor = torch.randn(1).numpy()
    stride_val = 2
    padding_val = 1
    output_padding_val = 1
    groups_val = 1
    dilation_val = 1
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(2, 2, 8, 8).numpy()
    weight_tensor = torch.randn(2, 1, 4, 4).numpy()
    bias_tensor = torch.randn(1).numpy()
    stride_val = 1
    padding_val = 2
    output_padding_val = 0
    groups_val = 1
    dilation_val = 2
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(1, 4, 12, 12).numpy()
    weight_tensor = torch.randn(4, 1, 3, 3).numpy()
    bias_tensor = torch.randn(1).numpy()
    stride_val = 3
    padding_val = 0
    output_padding_val = 2
    groups_val = 1
    dilation_val = 1
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5
    input_tensor = torch.randn(1, 2, 6, 6).numpy()
    weight_tensor = torch.randn(2, 2, 2, 2).numpy()
    bias_tensor = torch.randn(2).numpy()
    stride_val = 2
    padding_val = 1
    output_padding_val = 0
    groups_val = 1
    dilation_val = 1
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "output_padding": output_padding_val,
        "groups": groups_val,
        "dilation": dilation_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv_transpose2d_1"] = conv_transpose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv_transpose2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv_transpose2d_1'.")

check_valid('torch.nn.functional.conv_transpose2d', generated_inputs['torch.nn.functional.conv_transpose2d_1'], lib="torch", suffix=1)
