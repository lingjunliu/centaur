
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.randn(1, 3, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(5, 3, 5, 5).astype(np.float32)
    bias_tensor = np.random.randn(5).astype(np.float32)
    stride_val = (1, 1)
    padding_val = 0
    dilation_val = 1
    groups_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "dilation": dilation_val,
        "groups": groups_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.randn(2, 4, 16, 16).astype(np.float32)
    weight_tensor = np.random.randn(8, 4, 3, 3).astype(np.float32)
    bias_tensor = np.random.randn(8).astype(np.float32)
    stride_val = (2, 2)
    padding_val = 1
    dilation_val = 1
    groups_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "dilation": dilation_val,
        "groups": groups_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randn(1, 1, 64, 64).astype(np.float32)
    weight_tensor = np.random.randn(3, 1, 7, 7).astype(np.float32)
    bias_tensor = np.random.randn(3).astype(np.float32)
    stride_val = (3, 3)
    padding_val = 2
    dilation_val = 1
    groups_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "dilation": dilation_val,
        "groups": groups_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.randn(4, 2, 28, 28).astype(np.float32)
    weight_tensor = np.random.randn(6, 2, 4, 4).astype(np.float32)
    bias_tensor = np.random.randn(6).astype(np.float32)
    stride_val = (1, 2)
    padding_val = 0
    dilation_val = 2
    groups_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "dilation": dilation_val,
        "groups": groups_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (no bias)
    input_tensor = np.random.randn(1, 3, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(5, 3, 5, 5).astype(np.float32)
    bias_tensor = None
    stride_val = (1, 1)
    padding_val = 0
    dilation_val = 1
    groups_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "dilation": dilation_val,
        "groups": groups_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (dilation > 1)
    input_tensor = np.random.randn(1, 3, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(5, 3, 5, 5).astype(np.float32)
    bias_tensor = np.random.randn(5).astype(np.float32)
    stride_val = (1, 1)
    padding_val = 0
    dilation_val = 2
    groups_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "dilation": dilation_val,
        "groups": groups_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (padding > 0)
    input_tensor = np.random.randn(1, 3, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(5, 3, 5, 5).astype(np.float32)
    bias_tensor = np.random.randn(5).astype(np.float32)
    stride_val = (1, 1)
    padding_val = 2
    dilation_val = 1
    groups_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "dilation": dilation_val,
        "groups": groups_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (different stride values)
    input_tensor = np.random.randn(1, 3, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(5, 3, 5, 5).astype(np.float32)
    bias_tensor = np.random.randn(5).astype(np.float32)
    stride_val = (2, 3)
    padding_val = 1
    dilation_val = 1
    groups_val = 1

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "dilation": dilation_val,
        "groups": groups_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (groups > 1, adjusted weight shape)
    input_tensor = np.random.randn(1, 4, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(8, 1, 3, 3).astype(np.float32)
    bias_tensor = np.random.randn(8).astype(np.float32)
    stride_val = (1, 1)
    padding_val = 1
    dilation_val = 1
    groups_val = 4

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "dilation": dilation_val,
        "groups": groups_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (groups > 1, adjusted weight shape, no bias)
    input_tensor = np.random.randn(1, 4, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(8, 1, 3, 3).astype(np.float32)
    bias_tensor = None
    stride_val = (1, 1)
    padding_val = 1
    dilation_val = 1
    groups_val = 4

    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": stride_val,
        "padding": padding_val,
        "dilation": dilation_val,
        "groups": groups_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv2d_2"] = conv2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv2d_2'.")

check_valid('torch.nn.functional.conv2d', generated_inputs['torch.nn.functional.conv2d_2'], lib="torch", suffix=2)
