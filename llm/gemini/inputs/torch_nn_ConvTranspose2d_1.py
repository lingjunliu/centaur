
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def conv_transpose2d_inputs():
    list_of_inputs = []

    def get_output_shape(input_shape, p):
        N, C_out, H_in, W_in = input_shape[0], p['out_channels'], input_shape[2], input_shape[3]
        H_out = (H_in - 1) * p['stride'][0] - 2 * p['padding'][0] + p['dilation'][0] * (p['kernel_size'][0] - 1) + p['output_padding'][0] + 1
        W_out = (W_in - 1) * p['stride'][1] - 2 * p['padding'][1] + p['dilation'][1] * (p['kernel_size'][1] - 1) + p['output_padding'][1] + 1
        return (N, C_out, int(H_out), int(W_out))

    # Input 1: Basic case with stride > 1
    params1 = {
        'in_channels': 16, 'out_channels': 32, 'kernel_size': (3, 3),
        'stride': (2, 2), 'padding': (1, 1), 'output_padding': (1, 1),
        'groups': 1, 'bias': True, 'dilation': (1, 1),
        'padding_mode': 'zeros', 'dtype': torch.float32
    }
    input_shape1 = (1, 16, 12, 12)
    input_dict1 = {
        **params1,
        'input': np.random.randn(*input_shape1).astype(np.float32),
        'output_size': get_output_shape(input_shape1, params1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Non-square kernel, no bias, float64
    params2 = {
        'in_channels': 8, 'out_channels': 16, 'kernel_size': (3, 5),
        'stride': (2, 1), 'padding': (1, 2), 'output_padding': (0, 0),
        'groups': 1, 'bias': False, 'dilation': (1, 1),
        'padding_mode': 'zeros', 'dtype': torch.float64
    }
    input_shape2 = (4, 8, 20, 30)
    input_dict2 = {
        **params2,
        'input': np.random.randn(*input_shape2).astype(np.float64),
        'output_size': get_output_shape(input_shape2, params2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Grouped convolution
    params3 = {
        'in_channels': 16, 'out_channels': 32, 'kernel_size': (4, 4),
        'stride': (2, 2), 'padding': (1, 1), 'output_padding': (0, 0),
        'groups': 4, 'bias': True, 'dilation': (1, 1),
        'padding_mode': 'zeros', 'dtype': torch.float32
    }
    input_shape3 = (2, 16, 8, 8)
    input_dict3 = {
        **params3,
        'input': np.random.randn(*input_shape3).astype(np.float32),
        'output_size': get_output_shape(input_shape3, params3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Depthwise-like convolution
    params4 = {
        'in_channels': 8, 'out_channels': 8, 'kernel_size': (3, 3),
        'stride': (1, 1), 'padding': (1, 1), 'output_padding': (0, 0),
        'groups': 8, 'bias': False, 'dilation': (1, 1),
        'padding_mode': 'zeros', 'dtype': torch.float32
    }
    input_shape4 = (1, 8, 16, 16)
    input_dict4 = {
        **params4,
        'input': np.random.randn(*input_shape4).astype(np.float32),
        'output_size': get_output_shape(input_shape4, params4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Dilation > 1
    params5 = {
        'in_channels': 3, 'out_channels': 10, 'kernel_size': (3, 3),
        'stride': (1, 1), 'padding': (2, 2), 'output_padding': (0, 0),
        'groups': 1, 'bias': True, 'dilation': (2, 2),
        'padding_mode': 'zeros', 'dtype': torch.float32
    }
    input_shape5 = (1, 3, 28, 28)
    input_dict5 = {
        **params5,
        'input': np.random.randn(*input_shape5).astype(np.float32),
        'output_size': get_output_shape(input_shape5, params5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Valid padding_mode
    params6 = {
        'in_channels': 1, 'out_channels': 1, 'kernel_size': (5, 5),
        'stride': (2, 2), 'padding': (2, 2), 'output_padding': (1, 1),
        'groups': 1, 'bias': True, 'dilation': (1, 1),
        'padding_mode': 'zeros', 'dtype': torch.float32
    }
    input_shape6 = (1, 1, 7, 7)
    input_dict6 = {
        **params6,
        'input': np.random.randn(*input_shape6).astype(np.float32),
        'output_size': get_output_shape(input_shape6, params6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Valid padding_mode
    params7 = {
        'in_channels': 4, 'out_channels': 8, 'kernel_size': (3, 3),
        'stride': (1, 1), 'padding': (1, 1), 'output_padding': (0, 0),
        'groups': 2, 'bias': False, 'dilation': (1, 1),
        'padding_mode': 'zeros', 'dtype': torch.float32
    }
    input_shape7 = (1, 4, 32, 32)
    input_dict7 = {
        **params7,
        'input': np.random.randn(*input_shape7).astype(np.float32),
        'output_size': get_output_shape(input_shape7, params7)
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Large batch size and groups
    params8 = {
        'in_channels': 8, 'out_channels': 16, 'kernel_size': (3, 3),
        'stride': (2, 2), 'padding': (1, 1), 'output_padding': (1, 1),
        'groups': 4, 'bias': True, 'dilation': (1, 1),
        'padding_mode': 'zeros', 'dtype': torch.float32
    }
    input_shape8 = (32, 8, 5, 5)
    input_dict8 = {
        **params8,
        'input': np.random.randn(*input_shape8).astype(np.float32),
        'output_size': get_output_shape(input_shape8, params8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Large kernel, no padding
    params9 = {
        'in_channels': 1, 'out_channels': 1, 'kernel_size': (7, 7),
        'stride': (3, 3), 'padding': (0, 0), 'output_padding': (0, 0),
        'groups': 1, 'bias': True, 'dilation': (1, 1),
        'padding_mode': 'zeros', 'dtype': torch.float32
    }
    input_shape9 = (1, 1, 5, 5)
    input_dict9 = {
        **params9,
        'input': np.random.randn(*input_shape9).astype(np.float32),
        'output_size': get_output_shape(input_shape9, params9)
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Complex combination of parameters with valid padding_mode
    params10 = {
        'in_channels': 6, 'out_channels': 12, 'kernel_size': (4, 2),
        'stride': (2, 3), 'padding': (1, 2), 'output_padding': (1, 0),
        'groups': 3, 'bias': True, 'dilation': (2, 1),
        'padding_mode': 'zeros', 'dtype': torch.float32
    }
    input_shape10 = (5, 6, 10, 15)
    input_dict10 = {
        **params10,
        'input': np.random.randn(*input_shape10).astype(np.float32),
        'output_size': get_output_shape(input_shape10, params10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["torch.nn.ConvTranspose2d_1"] = conv_transpose2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.ConvTranspose2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.ConvTranspose2d_1'.")

check_valid('torch.nn.ConvTranspose2d', generated_inputs['torch.nn.ConvTranspose2d_1'], lib="torch", suffix=1)
