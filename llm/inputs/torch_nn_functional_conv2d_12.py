
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv2d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(1, 3, 32, 32).astype(np.float32)
    weight_tensor = np.random.rand(5, 3, 5, 5).astype(np.float32)
    bias_tensor = np.random.rand(5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1),
        "padding": "valid",
        "dilation": (1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(2, 4, 64, 64).astype(np.float32)
    weight_tensor = np.random.rand(8, 4, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(8).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1),
        "padding": "valid",
        "dilation": (1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(1, 16, 128, 128).astype(np.float32)
    weight_tensor = np.random.rand(32, 8, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(32).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1),
        "padding": "valid",
        "dilation": (2, 2),
        "groups": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(4, 3, 28, 28).astype(np.float32)
    weight_tensor = np.random.rand(10, 3, 5, 5).astype(np.float32)
    bias_tensor = np.random.rand(10).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1),
        "padding": "valid",
        "dilation": (1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(1, 64, 256, 256).astype(np.float32)
    weight_tensor = np.random.rand(128, 64, 1, 1).astype(np.float32)
    bias_tensor = np.random.rand(128).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (2, 2),
        "padding": 0,
        "dilation": (1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(2, 3, 16, 16).astype(np.float32)
    weight_tensor = np.random.rand(5, 3, 3, 3).astype(np.float32)
    bias_tensor = None
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1),
        "padding": "valid",
        "dilation": (1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(1, 4, 32, 32).astype(np.float32)
    weight_tensor = np.random.rand(8, 2, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(8).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1),
        "padding": "valid",
        "dilation": (1, 1),
        "groups": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 3, 64, 64).astype(np.float32)
    weight_tensor = np.random.rand(10, 3, 7, 7).astype(np.float32)
    bias_tensor = np.random.rand(10).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1),
        "padding": "valid",
        "dilation": (2, 2),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(4, 1, 128, 128).astype(np.float32)
    weight_tensor = np.random.rand(5, 1, 3, 3).astype(np.float32)
    bias_tensor = None
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1),
        "padding": "valid",
        "dilation": (1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 3, 256, 256).astype(np.float32)
    weight_tensor = np.random.rand(10, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(10).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (2, 2),
        "padding": 0,
        "dilation": (1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv2d_12"] = conv2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv2d_12' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv2d_12'.")

check_valid('torch.nn.functional.conv2d', generated_inputs['torch.nn.functional.conv2d_12'], lib="torch", suffix=12)
