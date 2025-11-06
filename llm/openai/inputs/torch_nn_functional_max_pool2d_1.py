
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def max_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = torch.arange(16.0, dtype=torch.float32).reshape(1, 1, 4, 4).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = torch.randn(3, 5, 5, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = torch.randn(2, 3, 7, 7, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = torch.randn(1, 2, 6, 8, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = torch.randn(4, 1, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 5,
        "stride": 5,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = torch.randn(1, 8, 3, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = torch.full((1, 1, 1, 10), 3.14, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 1,
        "stride": 3,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = torch.randn(2, 2, 9, 5, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 3,
        "stride": 3,
        "padding": 1,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_arr = torch.randn(2, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 4,
        "stride": 4,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = torch.randn(3, 3, 15, 15, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 2,
        "stride": 3,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (fixed padding)
    input_arr = torch.randn(1, 5, 7, 9, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 3,
        "stride": 4,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = torch.randn(1, 1, 7, 8, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13
    input_arr = torch.randn(1, 1, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 2,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14
    input_arr = torch.randn(5, 2, 12, 7, dtype=torch.float32).numpy()
    input_dict = {
        "input": input_arr,
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool2d_1"] = max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool2d_1'.")


check_valid('torch.nn.functional.max_pool2d', generated_inputs['torch.nn.functional.max_pool2d_1'], lib="torch", suffix=1)
