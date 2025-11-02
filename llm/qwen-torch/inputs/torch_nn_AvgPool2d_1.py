
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def avgpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with integer kernel_size, stride, padding
    input = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Different kernel_size and stride values
    input = torch.randn(2, 3, 6, 6).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With ceil_mode true
    input = torch.randn(1, 2, 5, 5).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With count_include_pad false
    input = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - With divisor_override specified
    input = torch.randn(1, 1, 4, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 3,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Non-square kernel and stride
    input = torch.randn(1, 1, 5, 5).numpy()
    input_dict = {
        "kernel_size": (2, 3),
        "stride": (1, 2),
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - With padding and different kernel size
    input = torch.randn(1, 1, 6, 6).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Large kernel_size and stride values
    input = torch.randn(1, 1, 8, 8).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 3,
        "padding": 2,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Different input dimensions (3D)
    input = torch.randn(1, 5, 4).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - With negative padding (valid case)
    input = torch.randn(1, 1, 6, 6).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input
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
