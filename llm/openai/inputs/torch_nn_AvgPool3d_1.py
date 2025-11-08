
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def avgpool3d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(2, 3, 8, 10, 12).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 8,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(1, 1, 5, 5, 5, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 27,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3 (4D input: C, D, H, W)
    input_arr = torch.randn(4, 7, 9, 11).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 8,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input_arr = torch.randn(3, 2, 4, 6, 8).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = torch.randn(2, 1, 3, 4, 5).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 8,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6 (4D)
    input_arr = torch.randn(1, 10, 10, 3).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 27,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input_arr = torch.randn(1, 5, 9, 7, 7, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 4,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 27,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input_arr = torch.randn(4, 3, 2, 2, 2).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 8,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = torch.randn(1, 2, 15, 15, 15).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 5,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 125,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10 (4D)
    input_arr = torch.randn(2, 6, 5, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 8,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = torch.randn(2, 2, 5, 4, 3).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 8,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12 (4D with small depth)
    input_arr = torch.randn(1, 1, 1, 10).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool3d_1"] = avgpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool3d_1'.")


check_valid('torch.nn.AvgPool3d', generated_inputs['torch.nn.AvgPool3d_1'], lib="torch", suffix=1)
