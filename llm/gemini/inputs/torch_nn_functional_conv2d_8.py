
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy
import torch.nn.functional as F

def conv2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.randn(1, 3, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(16, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.randn(16).astype(np.float32)
    stride_val = (1, 1)
    padding_val = 0
    dilation_val = (1, 1)
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
    input_tensor = np.random.randn(4, 3, 64, 64).astype(np.float32)
    weight_tensor = np.random.randn(8, 3, 5, 5).astype(np.float32)
    bias_tensor = np.random.randn(8).astype(np.float32)
    stride_val = (2, 2)
    padding_val = 2
    dilation_val = (1, 1)
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
    input_tensor = np.random.randn(1, 3, 128, 128).astype(np.float32)
    weight_tensor = np.random.randn(32, 3, 7, 7).astype(np.float32)
    bias_tensor = np.random.randn(32).astype(np.float32)
    stride_val = (1, 1)
    padding_val = 3
    dilation_val = (2, 2)
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
    input_tensor = np.random.randn(2, 6, 256, 256).astype(np.float32)
    weight_tensor = np.random.randn(12, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.randn(12).astype(np.float32)
    stride_val = (2, 2)
    padding_val = 1
    dilation_val = (1, 1)
    groups_val = 2
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
    
    # Input 5
    input_tensor = np.random.randn(1, 4, 16, 16).astype(np.float32)
    weight_tensor = np.random.randn(8, 2, 3, 3).astype(np.float32)
    bias_tensor = np.random.randn(8).astype(np.float32)
    stride_val = (1, 1)
    padding_val = 1
    dilation_val = (1, 1)
    groups_val = 2
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

    # Input 6
    input_tensor = np.random.randn(1, 1, 64, 64).astype(np.float32)
    weight_tensor = np.random.randn(4, 1, 5, 5).astype(np.float32)
    bias_tensor = np.random.randn(4).astype(np.float32)
    stride_val = (2, 2)
    padding_val = 2
    dilation_val = (1, 1)
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

    # Input 7
    input_tensor = np.random.randn(2, 8, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(16, 4, 3, 3).astype(np.float32)
    bias_tensor = np.random.randn(16).astype(np.float32)
    stride_val = (1, 1)
    padding_val = 1
    dilation_val = (1, 1)
    groups_val = 2
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
    
    # Input 8, no bias
    input_tensor = np.random.randn(1, 3, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(16, 3, 3, 3).astype(np.float32)
    bias_tensor = None
    stride_val = (1, 1)
    padding_val = 0
    dilation_val = (1, 1)
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

    # Input 9, different dilation
    input_tensor = np.random.randn(1, 3, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(16, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.randn(16).astype(np.float32)
    stride_val = (1, 1)
    padding_val = 0
    dilation_val = (2, 2)
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

    # Input 10, different stride and padding
    input_tensor = np.random.randn(1, 3, 32, 32).astype(np.float32)
    weight_tensor = np.random.randn(16, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.randn(16).astype(np.float32)
    stride_val = (2, 2)
    padding_val = 1
    dilation_val = (1, 1)
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

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv2d_8"] = conv2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv2d_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv2d_8'.")

check_valid('torch.nn.functional.conv2d', generated_inputs['torch.nn.functional.conv2d_8'], lib="torch", suffix=8)
