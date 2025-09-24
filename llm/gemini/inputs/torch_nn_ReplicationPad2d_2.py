
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def replicationpad2d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with 4D float tensor and asymmetric padding
    input_dict_1 = {
        'padding': (1, 2, 3, 4),
        'input': torch.randn(1, 1, 3, 3).numpy().astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Symmetric padding on a 4D float tensor
    input_dict_2 = {
        'padding': (2, 2, 2, 2),
        'input': torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D input tensor (C, H, W)
    input_dict_3 = {
        'padding': (1, 1, 1, 1),
        'input': torch.randn(3, 5, 5).numpy().astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 4D integer tensor
    input_dict_4 = {
        'padding': (1, 0, 1, 0),
        'input': torch.arange(16, dtype=torch.int32).reshape(1, 1, 4, 4).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Zero padding (output shape should be same as input)
    input_dict_5 = {
        'padding': (0, 0, 0, 0),
        'input': torch.randn(2, 3, 4, 5).numpy().astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Large padding values on a small tensor
    input_dict_6 = {
        'padding': (5, 5, 5, 5),
        'input': torch.ones(1, 1, 2, 2).numpy().astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Multiple channels (like an RGB image)
    input_dict_7 = {
        'padding': (1, 1, 2, 2),
        'input': torch.randn(1, 3, 10, 10).numpy().astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Multiple batches
    input_dict_8 = {
        'padding': (2, 1, 2, 1),
        'input': torch.randn(4, 1, 5, 5).numpy().astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Input with a dimension of size 1 (single row)
    input_dict_9 = {
        'padding': (1, 1, 1, 1),
        'input': torch.randn(1, 1, 1, 5).numpy().astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Input with a dimension of size 1 (single column)
    input_dict_10 = {
        'padding': (2, 2, 0, 0),
        'input': torch.randn(1, 1, 5, 1).numpy().astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Double precision float tensor
    input_dict_11 = {
        'padding': (1, 1, 1, 1),
        'input': torch.randn(1, 2, 3, 4, dtype=torch.float64).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: 3D input tensor with different padding on each side
    input_dict_12 = {
        'padding': (3, 2, 1, 0),
        'input': torch.randn(2, 6, 6).numpy().astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["torch.nn.ReplicationPad2d_2"] = replicationpad2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ReplicationPad2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ReplicationPad2d_2'.")

check_valid('torch.nn.ReplicationPad2d', generated_inputs['torch.nn.ReplicationPad2d_2'], lib="torch", suffix=2)
