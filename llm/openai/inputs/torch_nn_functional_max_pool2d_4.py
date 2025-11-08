
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def max_pool2d_inputs():
    list_of_inputs = []

    x1 = torch.arange(0, 16, dtype=torch.float32).reshape(1, 1, 4, 4).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x1,
        "kernel_size": (2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }))

    x2 = torch.randn(2, 3, 7, 5, dtype=torch.float64).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x2,
        "kernel_size": (3, 3),
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }))

    x3 = (torch.randn(1, 2, 6, 9, dtype=torch.float32) * 5 - 2).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x3,
        "kernel_size": (2, 3),
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }))

    x4 = torch.randint(-50, 50, (4, 1, 8, 7)).to(dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x4,
        "kernel_size": (3, 2),
        "stride": 3,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }))

    x5 = torch.linspace(-1, 1, steps=100, dtype=torch.float32).reshape(1, 1, 10, 10).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x5,
        "kernel_size": (2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False
    }))

    x6 = torch.randn(1, 2, 5, 5, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x6,
        "kernel_size": (3, 3),
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True
    }))

    x7 = torch.randn(4, 1, 6, 6, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x7,
        "kernel_size": (4, 4),
        "stride": 3,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }))

    x8 = torch.randn(3, 8, 8, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x8,
        "kernel_size": (2, 2),
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False
    }))

    x9 = torch.randn(2, 2, 9, 7, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x9,
        "kernel_size": (3, 2),
        "stride": 1,
        "padding": 1,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False
    }))

    x10 = torch.randn(1, 4, 12, 12, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x10,
        "kernel_size": (2, 2),
        "stride": 1,
        "padding": 1,
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False
    }))

    x11 = (torch.randn(1, 1, 5, 4, dtype=torch.float32) - 0.5).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x11,
        "kernel_size": (2, 2),
        "stride": 3,
        "padding": 1,
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": True
    }))

    x12 = torch.randn(8, 16, 14, 14, dtype=torch.float32).numpy()
    list_of_inputs.append(copy.deepcopy({
        "input": x12,
        "kernel_size": (3, 3),
        "stride": 2,
        "padding": 1,
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False
    }))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool2d_4"] = max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool2d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool2d_4'.")


check_valid('torch.nn.functional.max_pool2d', generated_inputs['torch.nn.functional.max_pool2d_4'], lib="torch", suffix=4)
