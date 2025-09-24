
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with 3D input
    input_dict_1 = {
        'input': torch.arange(0, 10, dtype=torch.float32).reshape(1, 1, 10).numpy(),
        'kernel_size': 2,
        'stride': 2,
        'padding': 0,
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: With padding and count_include_pad=False
    input_dict_2 = {
        'input': torch.ones(1, 1, 5, dtype=torch.float32).numpy(),
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'ceil_mode': False,
        'count_include_pad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With stride > kernel_size and multiple channels
    input_dict_3 = {
        'input': torch.randn(2, 3, 16).numpy(),
        'kernel_size': 3,
        'stride': 4,
        'padding': 0,
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: With ceil_mode=True
    input_dict_4 = {
        'input': torch.arange(0, 7, dtype=torch.float32).reshape(1, 1, 7).numpy(),
        'kernel_size': 3,
        'stride': 3,
        'padding': 0,
        'ceil_mode': True,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Another standard case
    input_dict_5 = {
        'input': torch.ones(1, 2, 8, dtype=torch.float32).numpy(),
        'kernel_size': 4,
        'stride': 4,
        'padding': 0,
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: With padding
    input_dict_6 = {
        'input': torch.tensor([[[10., 20., 30., 40.]]], dtype=torch.float32).numpy(),
        'kernel_size': 2,
        'stride': 2,
        'padding': 1,
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: 2D input (batch, seq_len)
    input_dict_7 = {
        'input': torch.arange(0, 12, dtype=torch.float32).reshape(3, 4).numpy(),
        'kernel_size': 2,
        'stride': 1,
        'padding': 0,
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Complex case with all options set
    input_dict_8 = {
        'input': torch.randn(2, 3, 11).numpy(),
        'kernel_size': 4,
        'stride': 2,
        'padding': 2,
        'ceil_mode': True,
        'count_include_pad': False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large padding
    input_dict_9 = {
        'input': torch.ones(1, 1, 4).numpy(),
        'kernel_size': 3,
        'stride': 1,
        'padding': 2,
        'ceil_mode': True,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Larger kernel and stride
    input_dict_10 = {
        'input': torch.randn(1, 1, 20).numpy(),
        'kernel_size': 5,
        'stride': 5,
        'padding': 0,
        'ceil_mode': False,
        'count_include_pad': True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool1d_1"] = avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.avg_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool1d_1'.")

check_valid('torch.nn.functional.avg_pool1d', generated_inputs['torch.nn.functional.avg_pool1d_1'], lib="torch", suffix=1)
