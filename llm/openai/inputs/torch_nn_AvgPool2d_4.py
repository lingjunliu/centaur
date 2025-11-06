
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def avgpool2d_inputs():
    list_of_inputs = []

    input_arr = torch.randn(1, 1, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 3, 10, 15, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (1, 1),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(4, 16, 32, 32, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 2,
        "padding": (2, 2),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 25,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 8, 7, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": (0, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 7, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 4, 6, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": (1, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.ones(5, 1, 9, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 2, 5, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 4,
        "padding": (2, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 16,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(1, 64, 17, 31, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": (1, 1),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 7,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(2, 4, 6, 6, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 3,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(8, 12, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 3,
        "padding": (2, 2),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 25,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_arr = torch.randn(3, 5, 13, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 1,
        "padding": (1, 2),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 8,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_4"] = avgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_4'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_4'], lib="torch", suffix=4)
