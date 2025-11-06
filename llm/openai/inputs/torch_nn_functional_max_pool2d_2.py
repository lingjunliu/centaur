
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def max_pool2d_inputs():
    list_of_inputs = []

    input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[[-1.0, 0.5, 2.0],
                            [3.0, -4.0, 1.5],
                            [2.5, 0.0, -0.5]]]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 2, 5, 7, dtype=torch.float64).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (3, 2),
        "stride": (2, 3),
        "padding": (1, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.linspace(-1, 1, steps=49, dtype=torch.float32).reshape(1, 1, 7, 7).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.ones(1, 1, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(2, 3, 8, 8, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(3, 10, 5, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (3, 3),
        "stride": (3, 2),
        "padding": (1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = (torch.arange(30, dtype=torch.float32).reshape(1, 1, 5, 6) - 10.0).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.tensor([[[[0.0, -1.0, 2.0],
                            [3.0, -4.0, 5.0],
                            [6.0, 7.0, -8.0]],
                           [[-2.0, 1.0, -3.0],
                            [4.0, -5.0, 6.0],
                            [7.0, -8.0, 9.0]]]], dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (2, 2),
        "stride": (3, 3),
        "padding": (1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(1, 1, 7, 7, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (2, 2),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": 3,
        "return_indices": False,
        "ceil_mode": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = torch.randn(4, 2, 9, 11, dtype=torch.float32).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (4, 3),
        "stride": (2, 4),
        "padding": (1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = torch.arange(0, 3 * 6 * 6, dtype=torch.float32).reshape(3, 6, 6)
    input = base.flip(-1).numpy()
    input_dict = {
        "input": input,
        "kernel_size": (2, 2),
        "stride": (3, 2),
        "padding": (0, 1),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool2d_2"] = max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool2d_2'.")


check_valid('torch.nn.functional.max_pool2d', generated_inputs['torch.nn.functional.max_pool2d_2'], lib="torch", suffix=2)
