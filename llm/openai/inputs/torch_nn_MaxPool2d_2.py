
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy

def maxpool2d_inputs():
    list_of_inputs = []

    # 1
    input = torch.randn(1, 1, 4, 4, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input = torch.randn(20, 16, 50, 32, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 2),
        "stride": (2, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input = torch.randn(3, 10, 8, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4
    input = torch.randn(2, 4, 10, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input = torch.randn(1, 2, 7, 5, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (3, 3),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input = torch.randn(4, 3, 5, 5, dtype=torch.double).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7
    input = torch.randn(2, 2, 8, 6, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (4, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8
    input = torch.randn(1, 1, 6, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 1),
        "padding": (0, 0),
        "dilation": (1, 3),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9 (fixed padding to be valid)
    input = torch.randn(1, 1, 2, 2, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10 (use float32 to avoid potential CPU half support issues)
    input = torch.randn(1, 8, 9, 9, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2),
        "stride": (2, 2),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input = torch.randn(5, 12, 12, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 3),
        "stride": (2, 2),
        "padding": (1, 0),
        "dilation": (2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12
    input = torch.randn(3, 3, 13, 7, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (5, 4),
        "stride": (4, 3),
        "padding": (2, 1),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 13
    input = torch.randn(6, 5, 15, 10, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (3, 5),
        "stride": (3, 2),
        "padding": (1, 2),
        "dilation": (1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 14
    input = torch.randn(2, 3, 11, 11, dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (1, 1),
        "stride": (2, 3),
        "padding": (0, 0),
        "dilation": (1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool2d_2"] = maxpool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool2d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool2d_2'.")


check_valid('torch.nn.MaxPool2d', generated_inputs['torch.nn.MaxPool2d_2'], lib="torch", suffix=2)
