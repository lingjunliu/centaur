
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def avgpool2d_inputs():
    list_of_inputs = []
    
    kernel_size = 2
    stride = (2, 2)
    padding = 0
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
    
    kernel_size = 3
    stride = (1, 1)
    padding = 1
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
    
    kernel_size = 3
    stride = (2, 2)
    padding = 0
    ceil_mode = True
    count_include_pad = True
    divisor_override = None
    input_tensor = torch.randn(2, 4, 15, 15).numpy()
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
    
    kernel_size = 2
    stride = (2, 2)
    padding = 1
    ceil_mode = False
    count_include_pad = False
    divisor_override = None
    input_tensor = torch.randn(1, 1, 10, 10).numpy()
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
    
    kernel_size = 3
    stride = (3, 3)
    padding = 0
    ceil_mode = False
    count_include_pad = True
    divisor_override = 5
    input_tensor = torch.randn(4, 8, 12, 12).numpy()
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
    
    kernel_size = 4
    stride = (4, 4)
    padding = 0
    ceil_mode = False
    count_include_pad = True
    divisor_override = None
    input_tensor = torch.randn(8, 16, 32, 32).numpy()
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.AvgPool2d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.AvgPool2d_3'.")


check_valid('torch.nn.AvgPool2d', generated_inputs['torch.nn.AvgPool2d_3'], lib="torch", suffix=3)
