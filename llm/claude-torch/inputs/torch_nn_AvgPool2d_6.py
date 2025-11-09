
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def avgpool2d_inputs():
    list_of_inputs = []
    
    kernel_size = (2, 2)
    stride = 2
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    input_tensor = torch.randn(1, 3, 8, 8).numpy()
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    input_tensor = torch.randn(16, 32, 32).numpy()
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    kernel_size = (3, 2)
    stride = 2
    padding = (1, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    input_tensor = torch.randn(2, 4, 10, 10).numpy()
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    kernel_size = (2, 2)
    stride = 2
    padding = (0, 0)
    ceil_mode = True
    count_include_pad = True
    divisor_override = None
    input_tensor = torch.randn(1, 1, 7, 7).numpy()
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    kernel_size = (3, 3)
    stride = 1
    padding = (1, 1)
    ceil_mode = False
    count_include_pad = False
    divisor_override = None
    input_tensor = torch.randn(1, 8, 16, 16).numpy()
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    kernel_size = (2, 2)
    stride = 1
    padding = (0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = 2
    input_tensor = torch.randn(1, 3, 4, 4).numpy()
    input_dict = {
        "kernel_size": kernel_size

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_6'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_6'], lib="torch", suffix=6)
