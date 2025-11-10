
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import torch
import numpy as np
import copy

def max_pool1d_inputs():
    list_of_inputs = []
    
    input_tensor = np.random.randn(1, 3, 10).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "kernel_size": (2,),
        "stride": (2,),
        "padding": (0,),
        "dilation": (1,),
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(2, 4, 15).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "kernel_size": (3,),
        "stride": (1,),
        "padding": (1,),
        "dilation": (1,),
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(1, 2, 8).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "kernel_size": (2,),
        "stride": (2,),
        "padding": (0,),
        "dilation": (1,),
        "ceil_mode": False,
        "return_indices": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(1, 1, 7).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (0,),
        "dilation": (1,),
        "ceil_mode": True,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(1, 5, 20).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "kernel_size": (3,),
        "stride": (1,),
        "padding": (0,),
        "dilation": (2,),
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(8, 16, 32).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "kernel_size": (4,),
        "stride": (4,),
        "padding": (0,),
        "dilation": (1,),
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(3, 8, 25).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "kernel_size": (3,),
        "stride": (2,),
        "padding": (2,),
        "dilation": (2,),
        "ceil_mode": False,
        "return_indices": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.random.randn(1, 3, 12).astype(np.float32)
    input_dict = {
        "input": input_tensor,
        "kernel_size": (2,),
        "stride": (1,),
        "padding": (1,),
        "dilation": (1,),
        "ceil_mode": True,
        "return_indices": True
    }
    list_

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool1d_2'.")


check_valid('torch.nn.functional.max_pool1d', generated_inputs['torch.nn.functional.max_pool1d_2'], lib="torch", suffix=2)
