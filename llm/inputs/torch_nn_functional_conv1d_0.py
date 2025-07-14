
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.randn(2, 3, 10).astype(np.float32)
    weight_tensor = np.random.randn(5, 3, 2).astype(np.float32)
    bias_tensor = np.random.randn(5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": 1,
        "padding": "valid",
        "dilation": 1,
        "groups": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.randn(1, 16, 20).astype(np.float32)
    weight_tensor = np.random.randn(32, 16, 3).astype(np.float32)
    bias_tensor = np.random.randn(32).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": 1,
        "padding": "valid",
        "dilation": 1,
        "groups": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.randn(4, 8, 30).astype(np.float32)
    weight_tensor = np.random.randn(16, 8, 4).astype(np.float32)
    bias_tensor = np.random.randn(16).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": 1,
        "padding": 2,
        "dilation": 2,
        "groups": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.randn(1, 4, 15).astype(np.float32)
    weight_tensor = np.random.randn(8, 1, 3).astype(np.float32)
    bias_tensor = np.random.randn(8).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.randn(3, 32, 40).astype(np.float32)
    weight_tensor = np.random.randn(64, 32, 5).astype(np.float32)
    bias_tensor = np.random.randn(64).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": 1,
        "padding": 0,
        "dilation": 3,
        "groups": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: No bias
    input_tensor = np.random.randn(2, 3, 10).astype(np.float32)
    weight_tensor = np.random.randn(5, 3, 2).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": None,
        "stride": 1,
        "padding": "valid",
        "dilation": 1,
        "groups": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: groups > 1
    input_tensor = np.random.randn(1, 8, 20).astype(np.float32)
    weight_tensor = np.random.randn(16, 4, 3).astype(np.float32)
    bias_tensor = np.random.randn(16).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": 1,
        "padding": "valid",
        "dilation": 1,
        "groups": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: small input size
    input_tensor = np.random.randn(1, 1, 5).astype(np.float32)
    weight_tensor = np.random.randn(2, 1, 3).astype(np.float32)
    bias_tensor = np.random.randn(2).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": 1,
        "padding": "valid",
        "dilation": 1,
        "groups": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: dilation > 1, groups > 1
    input_tensor = np.random.randn(1, 16, 30).astype(np.float32)
    weight_tensor = np.random.randn(32, 8, 3).astype(np.float32)
    bias_tensor = np.random.randn(32).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": 1,
        "padding": 1,
        "dilation": 2,
        "groups": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large dilation, small kernel
    input_tensor = np.random.randn(1, 3, 20).astype(np.float32)
    weight_tensor = np.random.randn(5, 3, 1).astype(np.float32)
    bias_tensor = np.random.randn(5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": 1,
        "padding": "valid",
        "dilation": 5,
        "groups": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv1d"] = conv1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv1d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv1d'.")

check_valid('torch.nn.functional.conv1d', generated_inputs['torch.nn.functional.conv1d'], lib="torch", suffix=0)
