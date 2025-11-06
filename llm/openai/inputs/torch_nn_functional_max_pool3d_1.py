
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def max_pool3d_inputs():
    list_of_inputs = []

    input = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 5, 6, 7, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = (-1.5 * torch.ones(1, 2, 3, 3, 3, dtype=torch.float32)).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 1, 8, 8, 8, dtype=torch.double).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 2,
        "stride": 3,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.linspace(-3, 3, steps=1*4*10*5*5, dtype=torch.float32).reshape(1, 4, 10, 5, 5).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.rand(3, 2, 9, 9, 9, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 4,
        "stride": 4,
        "padding": 2,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor(list(range(1*1*2*3*4)), dtype=torch.float32).reshape(1, 1, 2, 3, 4).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 5, 3, 8, 2, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 1,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 7, 7, 7, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 5,
        "stride": 2,
        "padding": 2,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2, 6, 5, 4, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 5, 4, 3, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 3,
        "stride": 3,
        "padding": 1,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 4, 12, 9, 7, dtype=torch.float64).numpy()
    input_dict = {
        "input": input,
        "kernel_size": 3,
        "stride": 4,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool3d_1"] = max_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool3d_1'.")


check_valid('torch.nn.functional.max_pool3d', generated_inputs['torch.nn.functional.max_pool3d_1'], lib="torch", suffix=1)
