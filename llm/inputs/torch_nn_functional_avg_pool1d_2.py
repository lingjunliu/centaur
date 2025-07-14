
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = torch.randn(1, 1, 5).numpy()
    kernel_size1 = (2,)
    stride1 = (1,)
    padding1 = (0,)
    ceil_mode1 = False
    count_include_pad1 = True

    input_dict1 = {
        'input': input1,
        'kernel_size': kernel_size1,
        'stride': stride1,
        'padding': padding1,
        'ceil_mode': ceil_mode1,
        'count_include_pad': count_include_pad1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = torch.randn(1, 3, 10).numpy()
    kernel_size2 = (3,)
    stride2 = (2,)
    padding2 = (1,)
    ceil_mode2 = True
    count_include_pad2 = False

    input_dict2 = {
        'input': input2,
        'kernel_size': kernel_size2,
        'stride': stride2,
        'padding': padding2,
        'ceil_mode': ceil_mode2,
        'count_include_pad': count_include_pad2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = torch.randn(2, 4, 8).numpy()
    kernel_size3 = (4,)
    stride3 = (3,)
    padding3 = (2,)
    ceil_mode3 = False
    count_include_pad3 = True

    input_dict3 = {
        'input': input3,
        'kernel_size': kernel_size3,
        'stride': stride3,
        'padding': padding3,
        'ceil_mode': ceil_mode3,
        'count_include_pad': count_include_pad3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = torch.randn(1, 2, 15).numpy()
    kernel_size4 = (5,)
    stride4 = (4,)
    padding4 = (0,)
    ceil_mode4 = True
    count_include_pad4 = False

    input_dict4 = {
        'input': input4,
        'kernel_size': kernel_size4,
        'stride': stride4,
        'padding': padding4,
        'ceil_mode': ceil_mode4,
        'count_include_pad': count_include_pad4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = torch.randn(3, 1, 7).numpy()
    kernel_size5 = (1,)
    stride5 = (1,)
    padding5 = (0,)
    ceil_mode5 = False
    count_include_pad5 = True

    input_dict5 = {
        'input': input5,
        'kernel_size': kernel_size5,
        'stride': stride5,
        'padding': padding5,
        'ceil_mode': ceil_mode5,
        'count_include_pad': count_include_pad5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6
    input6 = torch.randn(1, 1, 20).numpy()
    kernel_size6 = (6,)
    stride6 = (5,)
    padding6 = (3,)
    ceil_mode6 = True
    count_include_pad6 = False

    input_dict6 = {
        'input': input6,
        'kernel_size': kernel_size6,
        'stride': stride6,
        'padding': padding6,
        'ceil_mode': ceil_mode6,
        'count_include_pad': count_include_pad6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7
    input7 = torch.randn(2, 2, 12).numpy()
    kernel_size7 = (7,)
    stride7 = (6,)
    padding7 = (4,)
    ceil_mode7 = False
    count_include_pad7 = True

    input_dict7 = {
        'input': input7,
        'kernel_size': kernel_size7,
        'stride': stride7,
        'padding': padding7,
        'ceil_mode': ceil_mode7,
        'count_include_pad': count_include_pad7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = torch.randn(1, 3, 9).numpy()
    kernel_size8 = (8,)
    stride8 = (7,)
    padding8 = (1,)
    ceil_mode8 = True
    count_include_pad8 = False

    input_dict8 = {
        'input': input8,
        'kernel_size': kernel_size8,
        'stride': stride8,
        'padding': padding8,
        'ceil_mode': ceil_mode8,
        'count_include_pad': count_include_pad8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = torch.randn(3, 1, 18).numpy()
    kernel_size9 = (9,)
    stride9 = (8,)
    padding9 = (2,)
    ceil_mode9 = False
    count_include_pad9 = True

    input_dict9 = {
        'input': input9,
        'kernel_size': kernel_size9,
        'stride': stride9,
        'padding': padding9,
        'ceil_mode': ceil_mode9,
        'count_include_pad': count_include_pad9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = torch.randn(1, 2, 6).numpy()
    kernel_size10 = (10,)
    stride10 = (9,)
    padding10 = (5,)
    ceil_mode10 = True
    count_include_pad10 = False

    input_dict10 = {
        'input': input10,
        'kernel_size': kernel_size10,
        'stride': stride10,
        'padding': padding10,
        'ceil_mode': ceil_mode10,
        'count_include_pad': count_include_pad10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.avg_pool1d_2"] = avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool1d_2'.")

check_valid('torch.nn.functional.avg_pool1d', generated_inputs['torch.nn.functional.avg_pool1d_2'], lib="torch", suffix=2)
