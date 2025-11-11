
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def avgpool2d_inputs():
    list_of_inputs = []
    
    input_dict = {
        "kernel_size": (3, 3),
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": torch.randn(1, 3, 32, 32).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (2, 2),
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": torch.randn(2, 16, 28, 28).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (3, 3),
        "stride": 2,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": None,
        "input": torch.randn(4, 8, 25, 25).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (2, 2),
        "stride": 1,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": None,
        "input": torch.randn(1, 1, 10, 10).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (4, 4),
        "stride": 4,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 8,
        "input": torch.randn(1, 64, 16, 16).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (3, 5),
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": torch.randn(8, 32, 50, 40).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (2, 2),
        "stride": 5,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": torch.randn(1, 3, 64, 64).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": torch.randn(1, 16, 7, 7).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "kernel_size": (5, 5),
        "stride": 3,
        "padding": 2,
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": None,
        "input": torch.randn(1,

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
