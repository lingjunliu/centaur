
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.random.randn(1, 1, 4, 4, 4).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": [2, 2, 2],
        "stride": int(2),
        "padding": int(0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": int(8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.random.randn(2, 3, 5, 5, 5).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": [3, 3, 3],
        "stride": int(1),
        "padding": int(1),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": int(27)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = np.random.randn(2, 6, 4, 3).astype(np.float64)
    input_dict = {
        "input": input_arr,
        "kernel_size": [2, 3, 1],
        "stride": int(2),
        "padding": int(0),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": int(6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = (np.arange(1*2*3*3*3, dtype=np.float64).reshape(1, 2, 3, 3, 3) - 10.0).astype(np.float64)
    input_dict = {
        "input": input_arr,
        "kernel_size": [1, 1, 1],
        "stride": int(1),
        "padding": int(0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": int(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_arr = (np.random.randn(1, 1, 4, 5, 6) * 0.5).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": [2, 2, 3],
        "stride": int(3),
        "padding": int(1),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": int(12)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = (np.ones((1, 3, 3, 3), dtype=np.float32) * -2.0).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": [3, 3, 3],
        "stride": int(1),
        "padding": int(0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": int(27)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = np.random.randn(4, 2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": [2, 2, 2],
        "stride": int(2),
        "padding": int(1),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": int(8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = (np.random.randn(1, 4, 7, 6, 5) - 0.5).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": [1, 2, 2],
        "stride": int(4),
        "padding": int(0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": int(4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (padding set to 0 to avoid kernel_size=1 conflict)
    input_arr = np.random.randn(3, 5, 4, 7).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": [2, 1, 2],
        "stride": int(1),
        "padding": int(0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": int(4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = np.random.randn(1, 2, 2, 2, 2).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": [2, 2, 2],
        "stride": int(1),
        "padding": int(0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": int(8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = (np.random.randn(2, 1, 7, 5, 4) * 2.0).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": [3, 2, 2],
        "stride": int(2),
        "padding": int(1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": int(12)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = (np.arange(5*2*2*2, dtype=np.float64).reshape(5, 2, 2, 2) - 5.0).astype(np.float64)
    input_dict = {
        "input": input_arr,
        "kernel_size": [1, 2, 2],
        "stride": int(2),
        "padding": int(0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": int(4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool3d_8"] = avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.avg_pool3d_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool3d_8'.")


check_valid('torch.nn.functional.avg_pool3d', generated_inputs['torch.nn.functional.avg_pool3d_8'], lib="torch", suffix=8)
