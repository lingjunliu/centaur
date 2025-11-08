
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool2d_inputs():
    list_of_inputs = []

    input = torch.randn(1, 3, 8, 8).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 7, 7).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 2, 5, 5).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 4, 9, 7).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 6, 6).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 1, 5, 5, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 12, 5).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.linspace(-1, 1, steps=36).reshape(1, 1, 4, 9).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 7, 5).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 3, 7, 7).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2, 4, 4).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.arange(0, 3*6*6, dtype=torch.float32).reshape(3, 6, 6).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_1"] = maxpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool2d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool2d_1'.")


check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_1'], lib="torch", suffix=1)
