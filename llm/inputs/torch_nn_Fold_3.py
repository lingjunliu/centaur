
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy
import math

def fold_inputs():
    list_of_inputs = []

    def calculate_l(output_size, kernel_size, dilation, padding, stride):
        # This helper calculates the L dimension for the input tensor based on the formula
        # L = product_d(floor((output_size[d] + 2*padding[d] - dilation[d]*(kernel_size[d]-1) - 1) / stride[d] + 1))
        # Since the signature specifies integer parameters, we assume they are symmetric for height and width.
        l_dim = math.floor(
            (output_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
        return l_dim * l_dim

    # Input 1: Basic case with stride=1, padding=0, dilation=1
    params_1 = {'output_size': 4, 'kernel_size': 2, 'dilation': 1, 'padding': 0, 'stride': 1}
    N, C = 1, 3
    L = calculate_l(**params_1) # L = 3*3 = 9
    input_tensor_1 = torch.randn(N, C * params_1['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_1, 'input': input_tensor_1}))

    # Input 2: With padding
    params_2 = {'output_size': 5, 'kernel_size': 3, 'dilation': 1, 'padding': 1, 'stride': 1}
    N, C = 2, 1
    L = calculate_l(**params_2) # L = 5*5 = 25
    input_tensor_2 = torch.randn(N, C * params_2['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_2, 'input': input_tensor_2}))

    # Input 3: With stride > 1
    params_3 = {'output_size': 8, 'kernel_size': 2, 'dilation': 1, 'padding': 0, 'stride': 2}
    N, C = 1, 4
    L = calculate_l(**params_3) # L = 4*4 = 16
    input_tensor_3 = torch.randn(N, C * params_3['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_3, 'input': input_tensor_3}))

    # Input 4: With dilation > 1
    params_4 = {'output_size': 10, 'kernel_size': 3, 'dilation': 2, 'padding': 0, 'stride': 1}
    N, C = 3, 2
    L = calculate_l(**params_4) # L = 6*6 = 36
    input_tensor_4 = torch.randn(N, C * params_4['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_4, 'input': input_tensor_4}))

    # Input 5: All parameters > 1
    params_5 = {'output_size': 7, 'kernel_size': 3, 'dilation': 2, 'padding': 2, 'stride': 2}
    N, C = 1, 1
    L = calculate_l(**params_5) # L = 4*4 = 16
    input_tensor_5 = torch.randn(N, C * params_5['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_5, 'input': input_tensor_5}))

    # Input 6: Larger kernel size
    params_6 = {'output_size': 10, 'kernel_size': 5, 'dilation': 1, 'padding': 0, 'stride': 1}
    N, C = 1, 1
    L = calculate_l(**params_6) # L = 6*6 = 36
    input_tensor_6 = torch.randn(N, C * params_6['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_6, 'input': input_tensor_6}))

    # Input 7: kernel_size = 1
    params_7 = {'output_size': 8, 'kernel_size': 1, 'dilation': 1, 'padding': 0, 'stride': 1}
    N, C = 4, 8
    L = calculate_l(**params_7) # L = 8*8 = 64
    input_tensor_7 = torch.randn(N, C * params_7['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_7, 'input': input_tensor_7}))

    # Input 8: Larger stride and padding
    params_8 = {'output_size': 16, 'kernel_size': 4, 'dilation': 1, 'padding': 2, 'stride': 4}
    N, C = 2, 2
    L = calculate_l(**params_8) # L = 5*5 = 25
    input_tensor_8 = torch.randn(N, C * params_8['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_8, 'input': input_tensor_8}))

    # Input 9: Stride causes fractional division (floor)
    params_9 = {'output_size': 6, 'kernel_size': 3, 'dilation': 1, 'padding': 0, 'stride': 2}
    N, C = 1, 3
    L = calculate_l(**params_9) # L = 2*2 = 4
    input_tensor_9 = torch.randn(N, C * params_9['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_9, 'input': input_tensor_9}))

    # Input 10: Another combination with dilation and padding
    params_10 = {'output_size': 9, 'kernel_size': 4, 'dilation': 2, 'padding': 1, 'stride': 1}
    N, C = 1, 1
    L = calculate_l(**params_10) # L = 5*5 = 25
    input_tensor_10 = torch.randn(N, C * params_10['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_10, 'input': input_tensor_10}))

    # Input 11: Larger values for all parameters
    params_11 = {'output_size': 32, 'kernel_size': 5, 'dilation': 3, 'padding': 4, 'stride': 2}
    N, C = 1, 1
    L = calculate_l(**params_11) # L = 14*14 = 196
    input_tensor_11 = torch.randn(N, C * params_11['kernel_size']**2, L).numpy()
    list_of_inputs.append(copy.deepcopy({**params_11, 'input': input_tensor_11}))

    return list_of_inputs

generated_inputs["torch.nn.Fold_3"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.Fold_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.Fold_3'.")

check_valid('torch.nn.Fold', generated_inputs['torch.nn.Fold_3'], lib="torch", suffix=3)
