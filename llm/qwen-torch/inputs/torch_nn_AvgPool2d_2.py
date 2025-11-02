
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def avgpool2d_inputs():
    list_of_inputs = []
    
    # Input 1 - Basic case with square kernel and stride
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
    
    # Input 2 - Non-square kernel and stride
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
    
    # Input 3 - With padding
    input = torch.randn(5, 4, 20, 20).numpy()
    kernel_size = (4, 4)
    stride = (1, 1)
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
    
    # Input 4 - With ceil_mode
    input = torch.randn(15, 32, 30, 30).numpy()
    kernel_size = (5, 5)
    stride = (3, 3)
    padding = (0, 0)
    ceil_mode = True
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
    
    # Input 5 - With divisor_override
    input = torch.randn(10, 8, 64, 64).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = 4
    
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
    
    # Input 6 - Negative values in input tensor
    input = torch.randn(3, 2, 10, 10).numpy()
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
    
    # Input 7 - Different dimensions (4D)
    input = torch.randn(1, 1, 100, 100).numpy()
    kernel_size = (7, 7)
    stride = (4, 4)
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
    
    # Input 8 - Large kernel size
    input = torch.randn(2, 16, 200, 200).numpy()
    kernel_size = (10, 10)
    stride = (5, 5)
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
    
    # Input 9 - With count_include_pad set to False
    input = torch.randn(5, 8, 10, 10).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
    padding = (0, 0)
    ceil_mode = False
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
    
    # Input 10 - With positive padding (valid)
    input = torch.randn(5, 8, 10, 10).numpy()
    kernel_size = (2, 2)
    stride = (1, 1)
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
