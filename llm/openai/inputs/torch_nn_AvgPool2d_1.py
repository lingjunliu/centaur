
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def avgpool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.randn(1, 1, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(2, 3, 7, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 9,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(3, 3, 5, 5, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 3,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (3D input)
    input_arr = torch.randn(3, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 4,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 16,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.randn(1, 2, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 5,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.randn(5, 4, 10, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 2,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.randn(2, 1, 1, 5, dtype=torch.float64).numpy()
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

    # Input 8
    input_arr = torch.randn(4, 6, 9, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 3,
        "padding": 2,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 7,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.randn(3, 3, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.randn(10, 10, 32, 32, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 3,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (3D input)
    input_arr = torch.randn(1, 5, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 2,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.randn(2, 3, 5, 1, dtype=torch.float64).numpy()
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

    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_1"] = avgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_1'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_1'], lib="torch", suffix=1)
