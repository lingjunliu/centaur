
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def maxpool3d_inputs():
    list_of_inputs = []
    
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_tensor = torch.randn(1, 1, 8, 8, 8).numpy()
    
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_tensor = torch.randn(16, 10, 10, 10).numpy()
    
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    return_indices = True
    ceil_mode = False
    input_tensor = torch.randn(2, 3, 16, 16, 16).numpy()
    
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = True
    input_tensor = torch.randn(4, 8, 15, 15, 15).numpy()
    
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    kernel_size = 3
    stride = 1
    padding = 2
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_tensor = torch.randn(1, 4, 12, 12, 12).numpy()
    
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 2
    return_indices = False
    ceil_mode = False
    input_tensor = torch.randn(3, 5, 20, 20, 20).numpy()
    
    input_dict = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode,
        "input": input_tensor
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_1'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_1'], lib="torch", suffix=1)
