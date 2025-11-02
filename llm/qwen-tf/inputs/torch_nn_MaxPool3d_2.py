
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def maxpool3d_inputs():
    list_of_inputs = []

    # 1
    input_arr = np.random.randn(1, 1, 5, 6, 7).astype(np.float32)
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 2 (4D input)
    input_arr = np.random.randn(3, 10, 12, 14).astype(np.float32)
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3
    input_arr = np.random.randn(2, 4, 9, 9, 9).astype(np.float64)
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 4 (fixed padding to 1 to satisfy constraint)
    input_arr = np.random.randn(1, 2, 8, 6, 6).astype(np.float32)
    input_dict = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": (2, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 5
    input_arr = np.random.randn(1, 1, 7, 5, 4).astype(np.float32)
    input_dict = {
        "kernel_size": 2,
        "stride": 3,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 6
    input_arr = np.random.randn(2, 3, 15, 15, 15).astype(np.float32)
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": (2, 2, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 7 (fixed padding to 0 because k=1)
    input_arr = np.random.randn(4, 2, 3, 3, 3).astype(np.float32)
    input_dict = {
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": (3, 3, 3),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 8 (4D input)
    input_arr = np.random.randn(1, 9, 8, 7).astype(np.float32)
    input_dict = {
        "kernel_size": 3,
        "stride": 3,
        "padding": 0,
        "dilation": (1, 2, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 9
    input_arr = np.random.randn(1, 5, 10, 12, 20).astype(np.float32)
    input_dict = {
        "kernel_size": 4,
        "stride": 4,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 10
    input_arr = np.random.randn(8, 3, 16, 16, 16).astype(np.float64)
    input_dict = {
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 11
    input_arr = np.random.randn(1, 1, 2, 2, 2).astype(np.float32)
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": (1, 1, 1),
        "return_indices": True,
        "ceil_mode": True,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 12 (fixed padding to 1 to satisfy constraint for dilation=1 dim)
    input_arr = np.random.randn(2, 1, 20, 10, 12).astype(np.float32)
    input_dict = {
        "kernel_size": 3,
        "stride": 2,
        "padding": 1,
        "dilation": (1, 3, 2),
        "return_indices": False,
        "ceil_mode": False,
        "input": input_arr
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.MaxPool3d_2"] = maxpool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.MaxPool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.MaxPool3d_2'.")


check_valid('torch.nn.MaxPool3d', generated_inputs['torch.nn.MaxPool3d_2'], lib="torch", suffix=2)
