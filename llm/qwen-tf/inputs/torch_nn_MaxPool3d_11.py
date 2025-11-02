
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def maxpool3d_inputs():
    list_of_inputs = []

    # 1
    input_arr = torch.randn(1, 1, 8, 8, 8).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2
    input_arr = torch.randn(2, 3, 10, 12, 14).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 3,
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3 (4D input: C, D, H, W)
    input_arr = torch.randn(4, 7, 9, 11).numpy()
    input_dict = {
        "kernel_size": (2, 3, 2),
        "stride": 2,
        "padding": (0, 1, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4 (dilation > 1)
    input_arr = torch.randn(4, 2, 5, 5, 5).numpy()
    input_dict = {
        "kernel_size": (3, 2, 2),
        "stride": 1,
        "padding": (0, 0, 0),
        "dilation": 2,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5 (mixed padding)
    input_arr = torch.randn(1, 1, 6, 7, 8).numpy()
    input_dict = {
        "kernel_size": (4, 3, 2),
        "stride": 2,
        "padding": (1, 0, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6 (ceil_mode and return_indices True)
    input_arr = torch.randn(3, 5, 9, 9, 9).numpy()
    input_dict = {
        "kernel_size": (3, 3, 3),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7 (larger padding on depth)
    input_arr = torch.randn(2, 1, 15, 13, 11).numpy()
    input_dict = {
        "kernel_size": (5, 3, 3),
        "stride": 3,
        "padding": (2, 1, 1),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8 (4D input with ceil_mode True)
    input_arr = torch.randn(2, 4, 6, 8).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9 (dilation 2 with padding and larger stride)
    input_arr = torch.randn(1, 7, 16, 12, 9).numpy()
    input_dict = {
        "kernel_size": (4, 4, 3),
        "stride": 4,
        "padding": (1, 2, 1),
        "dilation": 2,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10 (wide kernels in H and W)
    input_arr = torch.randn(5, 3, 20, 10, 5).numpy()
    input_dict = {
        "kernel_size": (2, 5, 5),
        "stride": 5,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11 (small input with padding to allow pooling)
    input_arr = torch.randn(2, 2, 3, 3, 3).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 1,
        "padding": (1, 1, 1),
        "dilation": 1,
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12 (very small input, ceil_mode True)
    input_arr = torch.tensor([[[[[1.0, -1.0],
                                 [2.0, -2.0]],
                                [[-3.0, 3.0],
                                 [4.0, -4.0]]]]], dtype=torch.float32).numpy()
    input_dict = {
        "kernel_size": (2, 2, 2),
        "stride": 2,
        "padding": (0, 0, 0),
        "dilation": 1,
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_11"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_11' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_11'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_11'], lib="torch", suffix=11)
