
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def maxpool3d_inputs():
    list_of_inputs = []

    inp = torch.randn(2, 3, 8, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 4, 7, 5, 9, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 1, 2),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(3, 2, 10, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 4,
        "stride": (3, 2, 2),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 9, 12, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2, 2),
        "padding": (1, 1, 0),
        "dilation": (2, 3, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(5, 3, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 2, 4, 5, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 1,
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (3, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 2, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": (1, 1, 0),
        "dilation": (3, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 3, 15, 20, 16, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 5,
        "stride": (3, 4, 2),
        "padding": (2, 0, 1),
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(4, 1, 10, 11, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (3, 2, 4),
        "padding": (1, 1, 1),
        "dilation": (6, 6, 6),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(16, 18, 22, 30, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (2, 3, 5),
        "padding": (0, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randint(-10, 10, (1, 2, 6, 6, 6), dtype=torch.int32).to(torch.float32).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 7, 6, 9, 10, dtype=torch.float64).numpy()
    input_dict = {
        "kernel_size": 3,
        "stride": (1, 2, 3),
        "padding": (1, 1, 1),
        "dilation": (3, 2, 5),
        "return_indices": True,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(2, 2, 7, 7, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "dilation": (3, 3, 3),
        "return_indices": False,
        "ceil_mode": False,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    inp = torch.randn(1, 1, 3, 3, 3, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": 2,
        "stride": (3, 3, 3),
        "padding": (1, 1, 1),
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": inp
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_8"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_8'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_8'], lib="torch", suffix=8)
