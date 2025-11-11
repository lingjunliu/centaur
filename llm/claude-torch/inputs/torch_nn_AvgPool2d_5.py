
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def avgpool2d_inputs():
    list_of_inputs = []
    
    input_tensor = torch.randn(1, 3, 32, 32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 16, 28, 28).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": (1, 1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 64, 15, 15).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(4, 8, 20, 20).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": (1, 1),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 32, 16, 16).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 2,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 1, 50, 50).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 5,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(2, 10, 24, 24).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 2,
        "padding": (1, 2),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = torch.randn(1, 5, 10, 10).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": (0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": None,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor =

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
