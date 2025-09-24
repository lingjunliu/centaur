
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np
import torch.nn.functional as F

def conv3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(2, 3, 10, 10, 10).astype(np.float32)
    weight_tensor = np.random.rand(5, 3, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(5).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (1, 1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 4, 8, 8, 8).astype(np.float32)
    weight_tensor = np.random.rand(8, 1, 2, 2, 2).astype(np.float32)
    bias_tensor = np.random.rand(8).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (1, 1, 1),
        "groups": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(4, 6, 12, 12, 12).astype(np.float32)
    weight_tensor = np.random.rand(9, 2, 4, 4, 4).astype(np.float32)
    bias_tensor = np.random.rand(9).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (2, 1, 1),
        "groups": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 3, 16, 16, 16).astype(np.float32)
    weight_tensor = np.random.rand(4, 3, 5, 5, 5).astype(np.float32)
    bias_tensor = np.random.rand(4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (1, 1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(2, 2, 9, 9, 9).astype(np.float32)
    weight_tensor = np.random.rand(4, 1, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (1, 1, 1),
        "groups": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(1, 1, 7, 7, 7).astype(np.float32)
    weight_tensor = np.random.rand(2, 1, 2, 2, 2).astype(np.float32)
    bias_tensor = np.random.rand(2).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (2, 2, 2),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.random.rand(3, 4, 11, 11, 11).astype(np.float32)
    weight_tensor = np.random.rand(8, 1, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(8).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (1, 2, 1),
        "groups": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(1, 3, 13, 13, 13).astype(np.float32)
    weight_tensor = np.random.rand(6, 1, 4, 4, 4).astype(np.float32)
    bias_tensor = np.random.rand(6).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (1, 1, 2),
        "groups": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(2, 1, 14, 14, 14).astype(np.float32)
    weight_tensor = np.random.rand(3, 1, 5, 5, 5).astype(np.float32)
    bias_tensor = np.random.rand(3).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (2, 1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(1, 2, 15, 15, 15).astype(np.float32)
    weight_tensor = np.random.rand(4, 1, 3, 3, 3).astype(np.float32)
    bias_tensor = np.random.rand(4).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (1, 2, 2),
        "groups": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_tensor = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    weight_tensor = np.random.rand(1, 1, 1, 1, 1).astype(np.float32)
    bias_tensor = np.random.rand(1).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "weight": weight_tensor,
        "bias": bias_tensor,
        "stride": (1, 1, 1),
        "padding": 'valid',
        "dilation": (1, 1, 1),
        "groups": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.conv3d"] = conv3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.conv3d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.conv3d'.")

check_valid('torch.nn.functional.conv3d', generated_inputs['torch.nn.functional.conv3d'], lib="torch", suffix=0)
