
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = np.random.rand(2, 3, 10, 10, 10).astype(np.float32)
    kernel_size1 = 3
    stride1 = (2, 2, 2)
    padding1 = (1, 1, 1)
    ceil_mode1 = False
    count_include_pad1 = True
    divisor_override1 = 1

    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "ceil_mode": ceil_mode1,
        "count_include_pad": count_include_pad1,
        "divisor_override": divisor_override1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2
    input2 = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    kernel_size2 = 2
    stride2 = (1, 1, 1)
    padding2 = (0, 0, 0)
    ceil_mode2 = True
    count_include_pad2 = False
    divisor_override2 = 2

    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "ceil_mode": ceil_mode2,
        "count_include_pad": count_include_pad2,
        "divisor_override": divisor_override2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3
    input3 = np.random.rand(4, 2, 7, 7, 7).astype(np.float32)
    kernel_size3 = 4
    stride3 = (3, 3, 3)
    padding3 = (2, 2, 2)
    ceil_mode3 = False
    count_include_pad3 = False
    divisor_override3 = 3

    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "ceil_mode": ceil_mode3,
        "count_include_pad": count_include_pad3,
        "divisor_override": divisor_override3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4
    input4 = np.random.rand(1, 1, 3, 3, 3).astype(np.float32)
    kernel_size4 = 1
    stride4 = (1, 1, 1)
    padding4 = (0, 0, 0)
    ceil_mode4 = True
    count_include_pad4 = True
    divisor_override4 = 1

    input_dict4 = {
        "input": input4,
        "kernel_size": kernel_size4,
        "stride": stride4,
        "padding": padding4,
        "ceil_mode": ceil_mode4,
        "count_include_pad": count_include_pad4,
        "divisor_override": divisor_override4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5
    input5 = np.random.rand(1, 2, 4, 4, 4).astype(np.float32)
    kernel_size5 = 2
    stride5 = (2, 2, 2)
    padding5 = (1, 1, 1)
    ceil_mode5 = False
    count_include_pad5 = True
    divisor_override5 = 1 # Modified from 0 to 1

    input_dict5 = {
        "input": input5,
        "kernel_size": kernel_size5,
        "stride": stride5,
        "padding": padding5,
        "ceil_mode": ceil_mode5,
        "count_include_pad": count_include_pad5,
        "divisor_override": divisor_override5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    # Input 6
    input6 = np.random.rand(3, 1, 6, 6, 6).astype(np.float32)
    kernel_size6 = 3
    stride6 = (1, 2, 1)
    padding6 = (0, 1, 0)
    ceil_mode6 = True
    count_include_pad6 = False
    divisor_override6 = 4

    input_dict6 = {
        "input": input6,
        "kernel_size": kernel_size6,
        "stride": stride6,
        "padding": padding6,
        "ceil_mode": ceil_mode6,
        "count_include_pad": count_include_pad6,
        "divisor_override": divisor_override6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))
    
    # Input 7
    input7 = np.random.rand(1, 1, 12, 12, 12).astype(np.float32)
    kernel_size7 = 5
    stride7 = (4, 4, 4)
    padding7 = (2, 2, 2)
    ceil_mode7 = False
    count_include_pad7 = True
    divisor_override7 = 1

    input_dict7 = {
        "input": input7,
        "kernel_size": kernel_size7,
        "stride": stride7,
        "padding": padding7,
        "ceil_mode": ceil_mode7,
        "count_include_pad": count_include_pad7,
        "divisor_override": divisor_override7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8
    input8 = np.random.rand(2, 2, 8, 8, 8).astype(np.float32)
    kernel_size8 = 4
    stride8 = (2, 1, 2)
    padding8 = (1, 0, 1)
    ceil_mode8 = True
    count_include_pad8 = False
    divisor_override8 = 2

    input_dict8 = {
        "input": input8,
        "kernel_size": kernel_size8,
        "stride": stride8,
        "padding": padding8,
        "ceil_mode": ceil_mode8,
        "count_include_pad": count_include_pad8,
        "divisor_override": divisor_override8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9
    input9 = np.random.rand(1, 3, 9, 9, 9).astype(np.float32)
    kernel_size9 = 3
    stride9 = (3, 3, 3)
    padding9 = (1, 1, 1)
    ceil_mode9 = False
    count_include_pad9 = True
    divisor_override9 = 3

    input_dict9 = {
        "input": input9,
        "kernel_size": kernel_size9,
        "stride": stride9,
        "padding": padding9,
        "ceil_mode": ceil_mode9,
        "count_include_pad": count_include_pad9,
        "divisor_override": divisor_override9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10
    input10 = np.random.rand(1, 1, 11, 11, 11).astype(np.float32)
    kernel_size10 = 5
    stride10 = (2, 2, 2)
    padding10 = (0, 0, 0)
    ceil_mode10 = True
    count_include_pad10 = False
    divisor_override10 = 1

    input_dict10 = {
        "input": input10,
        "kernel_size": kernel_size10,
        "stride": stride10,
        "padding": padding10,
        "ceil_mode": ceil_mode10,
        "count_include_pad": count_include_pad10,
        "divisor_override": divisor_override10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.avg_pool3d_4"] = avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool3d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool3d_4'.")

check_valid('torch.nn.functional.avg_pool3d', generated_inputs['torch.nn.functional.avg_pool3d_4'], lib="torch", suffix=4)
