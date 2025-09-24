
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import copy
import numpy as np

def fold_inputs():
    """
    Generates a list of valid inputs for the torch.nn.functional.fold function.
    """

    def _calculate_L(output_size, kernel_size, dilation, padding, stride):
        """Helper to calculate the L dimension for the input tensor."""
        H_out, W_out = output_size
        K_h, K_w = kernel_size
        D, P, S = dilation, padding, stride
        L_h = np.floor((H_out + 2 * P - D * (K_h - 1) - 1) / S + 1)
        L_w = np.floor((W_out + 2 * P - D * (K_w - 1) - 1) / S + 1)
        return int(L_h * L_w)

    list_of_inputs = []

    # Case 1: Basic batched case (N=1, C=1)
    params1 = {'output_size': (4, 5), 'kernel_size': (2, 2), 'dilation': 1, 'padding': 0, 'stride': 1}
    L1 = _calculate_L(**params1)
    input_shape1 = (1, 1 * params1['kernel_size'][0] * params1['kernel_size'][1], L1)
    input1 = torch.ones(input_shape1).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input1, **params1}))

    # Case 2: Basic unbatched case (3D output)
    params2 = {'output_size': (4, 4), 'kernel_size': (3, 3), 'dilation': 1, 'padding': 0, 'stride': 1}
    L2 = _calculate_L(**params2)
    C2 = 3
    input_shape2 = (C2 * params2['kernel_size'][0] * params2['kernel_size'][1], L2)
    input2 = torch.randn(input_shape2).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input2, **params2}))

    # Case 3: With padding
    params3 = {'output_size': (5, 5), 'kernel_size': (3, 3), 'dilation': 1, 'padding': 1, 'stride': 1}
    L3 = _calculate_L(**params3)
    N3, C3 = 2, 1
    input_shape3 = (N3, C3 * params3['kernel_size'][0] * params3['kernel_size'][1], L3)
    input3 = torch.ones(input_shape3).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input3, **params3}))

    # Case 4: With stride
    params4 = {'output_size': (10, 10), 'kernel_size': (2, 2), 'dilation': 1, 'padding': 0, 'stride': 2}
    L4 = _calculate_L(**params4)
    N4, C4 = 1, 3
    input_shape4 = (N4, C4 * params4['kernel_size'][0] * params4['kernel_size'][1], L4)
    input4 = torch.randn(input_shape4).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input4, **params4}))

    # Case 5: With dilation
    params5 = {'output_size': (10, 10), 'kernel_size': (3, 3), 'dilation': 2, 'padding': 0, 'stride': 1}
    L5 = _calculate_L(**params5)
    N5, C5 = 4, 1
    input_shape5 = (N5, C5 * params5['kernel_size'][0] * params5['kernel_size'][1], L5)
    input5 = torch.ones(input_shape5).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input5, **params5}))

    # Case 6: All parameters combined (batched)
    params6 = {'output_size': (13, 13), 'kernel_size': (3, 3), 'dilation': 2, 'padding': 2, 'stride': 3}
    L6 = _calculate_L(**params6)
    N6, C6 = 2, 2
    input_shape6 = (N6, C6 * params6['kernel_size'][0] * params6['kernel_size'][1], L6)
    input6 = torch.randn(input_shape6).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input6, **params6}))

    # Case 7: All parameters combined (unbatched)
    params7 = {'output_size': (13, 13), 'kernel_size': (3, 3), 'dilation': 2, 'padding': 2, 'stride': 3}
    L7 = _calculate_L(**params7)
    C7 = 1
    input_shape7 = (C7 * params7['kernel_size'][0] * params7['kernel_size'][1], L7)
    input7 = torch.ones(input_shape7).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input7, **params7}))

    # Case 8: Non-square kernel and output size
    params8 = {'output_size': (8, 10), 'kernel_size': (2, 3), 'dilation': 1, 'padding': 0, 'stride': 1}
    L8 = _calculate_L(**params8)
    N8, C8 = 1, 1
    input_shape8 = (N8, C8 * params8['kernel_size'][0] * params8['kernel_size'][1], L8)
    input8 = torch.randn(input_shape8).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input8, **params8}))

    # Case 9: Large output size
    params9 = {'output_size': (64, 64), 'kernel_size': (4, 4), 'dilation': 1, 'padding': 1, 'stride': 2}
    L9 = _calculate_L(**params9)
    N9, C9 = 1, 3
    input_shape9 = (N9, C9 * params9['kernel_size'][0] * params9['kernel_size'][1], L9)
    input9 = torch.ones(input_shape9).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input9, **params9}))

    # Case 10: Minimum valid size
    params10 = {'output_size': (1, 1), 'kernel_size': (1, 1), 'dilation': 1, 'padding': 0, 'stride': 1}
    L10 = _calculate_L(**params10)
    N10, C10 = 1, 1
    input_shape10 = (N10, C10 * params10['kernel_size'][0] * params10['kernel_size'][1], L10)
    input10 = torch.randn(input_shape10).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input10, **params10}))

    # Case 11: Non-1 dilation with padding
    params11 = {'output_size': (7, 7), 'kernel_size': (3, 3), 'dilation': 3, 'padding': 1, 'stride': 1}
    L11 = _calculate_L(**params11)
    N11, C11 = 1, 1
    input_shape11 = (N11, C11 * params11['kernel_size'][0] * params11['kernel_size'][1], L11)
    input11 = torch.ones(input_shape11).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input11, **params11}))

    # Case 12: Zero padding
    params12 = {'output_size': (8, 8), 'kernel_size': (3, 3), 'dilation': 1, 'padding': 0, 'stride': 1}
    L12 = _calculate_L(**params12)
    N12, C12 = 1, 5
    input_shape12 = (N12, C12 * params12['kernel_size'][0] * params12['kernel_size'][1], L12)
    input12 = torch.randn(input_shape12).numpy()
    list_of_inputs.append(copy.deepcopy({'input': input12, **params12}))
    
    return list_of_inputs

generated_inputs["torch.nn.functional.fold"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.fold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.fold'.")

check_valid('torch.nn.functional.fold', generated_inputs['torch.nn.functional.fold'], lib="torch", suffix=0)
