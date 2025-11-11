
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def avgpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with tuple parameters
    input = torch.randn(20, 16, 50, 32).numpy()
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2 - Non-square window
    input = torch.randn(10, 8, 64, 64).numpy()
    kernel_size = (3, 2)
    stride = (2, 1)
    padding = (1, 1)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3 - With padding and ceil_mode=True
    input = torch.randn(5, 4, 32, 32).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (1, 1)
    ceil_mode = True
    count_include_pad = False
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4 - With divisor_override
    input = torch.randn(1, 2, 16, 16).numpy()
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = 5
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 5 - Different dimensions
    input = torch.randn(3, 1, 8, 8).numpy()
    kernel_size = (4, 4)
    stride = (2, 2)
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6 - Large kernel size with small stride
    input = torch.randn(1, 3, 64, 64).numpy()
    kernel_size = (8, 8)
    stride = (2, 2)
    padding = (3, 3)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7 - Negative values in input tensor (valid for AvgPool2d)
    input = torch.randn(2, 3, 16, 16).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8 - Zero padding case
    input = torch.randn(4, 2, 32, 32).numpy()
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9 - Different input dimensions
    input = torch.randn(1, 2, 32, 32).numpy()
    kernel_size = (5, 5)
    stride = (3, 3)
    padding = (2, 2)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10 - With zero padding and different stride
    input = torch.randn(2, 4, 8, 8).numpy()
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (1, 1)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["torch.nn.AvgPool2d_2"] = avgpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_2'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_2'], lib="torch", suffix=2)
