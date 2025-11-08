
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def avgpool2d_inputs():
    list_of_inputs = []

    inp = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 4,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 10, 8).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": 2,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 3,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 7, 7).double().numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 9,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(4, 2, 15, 13).numpy()
    input_dict = {
        "kernel_size": (5, 3),
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 5,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 3, 10).numpy()
    input_dict = {
        "kernel_size": (3, 5),
        "stride": 3,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 5,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 5, 32, 32, dtype=torch.float16).to(torch.float16).numpy()
    input_dict = {
        "kernel_size": (7, 7),
        "stride": 4,
        "padding": 2,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 7,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 4, 9, 5).numpy()
    input_dict = {
        "kernel_size": (1, 3),
        "stride": 1,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 3,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(8, 20, 6).numpy()
    input_dict = {
        "kernel_size": (4, 2),
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 8,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 6, 6).numpy()
    input_dict = {
        "kernel_size": (6, 6),
        "stride": 6,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 6,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 1, 5, 5).numpy()
    input_dict = {
        "kernel_size": (2, 3),
        "stride": 1,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 6,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 12, 9).numpy()
    input_dict = {
        "kernel_size": (3, 4),
        "stride": 3,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 4,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(5, 7, 1, 1).numpy()
    input_dict = {
        "kernel_size": (1, 1),
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_5"] = avgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_5'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_5'], lib="torch", suffix=5)
