
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def fold_inputs():
    list_of_inputs = []

    # Input 1: Minimal valid input
    input1 = torch.randn(1, 3 * 5 * 5, 676).numpy()
    output_size1 = (30, 30)
    kernel_size1 = (5, 5)
    input_dict1 = {
        "input": input1,
        "output_size": output_size1,
        "kernel_size": kernel_size1,
        "dilation": 1,
        "padding": 0,
        "stride": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different output size
    input2 = torch.randn(1, 3 * 3 * 3, 324).numpy()
    output_size2 = (20, 20)
    kernel_size2 = (3, 3)
    input_dict2 = {
        "input": input2,
        "output_size": output_size2,
        "kernel_size": kernel_size2,
        "dilation": 1,
        "padding": 0,
        "stride": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Different kernel size
    input3 = torch.randn(1, 1 * 7 * 7, 1156).numpy()
    output_size3 = (40, 40)
    kernel_size3 = (7, 7)
    input_dict3 = {
        "input": input3,
        "output_size": output_size3,
        "kernel_size": kernel_size3,
        "dilation": 1,
        "padding": 0,
        "stride": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Different dilation
    input4 = torch.randn(1, 1 * 3 * 3, 2116).numpy()
    output_size4 = (50, 50)
    kernel_size4 = (3, 3)
    input_dict4 = {
        "input": input4,
        "output_size": output_size4,
        "kernel_size": kernel_size4,
        "dilation": 2,
        "padding": 0,
        "stride": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Different padding
    input5 = torch.randn(1, 1 * 5 * 5, 3600).numpy()
    output_size5 = (60, 60)
    kernel_size5 = (5, 5)
    input_dict5 = {
        "input": input5,
        "output_size": output_size5,
        "kernel_size": kernel_size5,
        "dilation": 1,
        "padding": 2,
        "stride": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Different stride
    input6 = torch.randn(1, 1 * 3 * 3, 1156).numpy()
    output_size6 = (70, 70)
    kernel_size6 = (3, 3)
    input_dict6 = {
        "input": input6,
        "output_size": output_size6,
        "kernel_size": kernel_size6,
        "dilation": 1,
        "padding": 0,
        "stride": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Batched input
    input7 = torch.randn(2, 1 * 5 * 5, 676).numpy()
    output_size7 = (30, 30)
    kernel_size7 = (5, 5)
    input_dict7 = {
        "input": input7,
        "output_size": output_size7,
        "kernel_size": kernel_size7,
        "dilation": 1,
        "padding": 0,
        "stride": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Different kernel_size (non-square)
    input8 = torch.randn(1, 1 * 3 * 5, 728).numpy()
    output_size8 = (30, 30)
    kernel_size8 = (3, 5)
    input_dict8 = {
        "input": input8,
        "output_size": output_size8,
        "kernel_size": kernel_size8,
        "dilation": 1,
        "padding": 0,
        "stride": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9: Different output_size (non-square)
    input9 = torch.randn(1, 1 * 3 * 3, 1600).numpy()
    output_size9 = (40, 50)
    kernel_size9 = (3, 3)
    input_dict9 = {
        "input": input9,
        "output_size": output_size9,
        "kernel_size": kernel_size9,
        "dilation": 1,
        "padding": 0,
        "stride": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

   # Input 10: Larger values for parameters
    input10 = torch.randn(1, 3 * 7 * 7, 49).numpy()
    output_size10 = (10, 10)
    kernel_size10 = (7, 7)
    input_dict10 = {
        "input": input10,
        "output_size": output_size10,
        "kernel_size": kernel_size10,
        "dilation": 3,
        "padding": 5,
        "stride": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.fold"] = fold_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.fold' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.fold'.")

check_valid('torch.nn.functional.fold', generated_inputs['torch.nn.functional.fold'], lib="torch", suffix=0)
