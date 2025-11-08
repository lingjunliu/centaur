
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_arr = np.arange(1, 1 + 1*1*4*4*4, dtype=np.float32).reshape(1, 1, 4, 4, 4)
    input_dict = {
        "input": input_arr,
        "kernel_size": 2,
        "stride": (2, 2, 2),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 8
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_arr = np.random.randn(2, 3, 5, 6, 7).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": 3,
        "stride": (1, 2, 3),
        "padding": (1, 1, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 27
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_arr = np.linspace(-5, 5, num=1*2*3*3*3, dtype=np.float64).reshape(1, 2, 3, 3, 3)
    input_dict = {
        "input": input_arr,
        "kernel_size": 1,
        "stride": (1, 1, 1),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_arr = np.random.randn(4, 1, 8, 4, 2).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": 2,
        "stride": (2, 2, 1),
        "padding": (0, 1, 0),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (fixed: kernel fits depth; zero padding)
    input_arr = np.random.uniform(-1.0, 1.0, size=(1, 1, 1, 5, 5)).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": 1,
        "stride": (1, 2, 2),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_arr = np.random.randn(2, 2, 6, 6, 6).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": 4,
        "stride": (2, 2, 2),
        "padding": (1, 1, 1),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_arr = np.random.randn(3, 4, 9, 7, 5).astype(np.float64)
    input_dict = {
        "input": input_arr,
        "kernel_size": 3,
        "stride": (3, 2, 2),
        "padding": (0, 1, 0),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_arr = np.random.uniform(-10, 10, size=(1, 5, 2, 9, 9)).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": 2,
        "stride": (1, 3, 3),
        "padding": (0, 0, 0),
        "ceil_mode": True,
        "count_include_pad": True,
        "divisor_override": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (fixed: kernel fits all dims)
    input_arr = np.random.randn(2, 1, 10, 10, 3).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": 3,
        "stride": (5, 5, 1),
        "padding": (0, 0, 1),
        "ceil_mode": False,
        "count_include_pad": False,
        "divisor_override": 9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_arr = np.random.randn(1, 1, 3, 4, 5).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": 3,
        "stride": (2, 1, 2),
        "padding": (0, 1, 0),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_arr = np.random.randn(1, 2, 7, 7, 7).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": 7,
        "stride": (7, 7, 7),
        "padding": (0, 0, 0),
        "ceil_mode": False,
        "count_include_pad": True,
        "divisor_override": 343
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_arr = np.random.uniform(-2, 2, size=(5, 3, 4, 3, 2)).astype(np.float32)
    input_dict = {
        "input": input_arr,
        "kernel_size": 2,
        "stride": (1, 1, 1),
        "padding": (1, 0, 1),
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool3d_4"] = avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.avg_pool3d_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool3d_4'.")


check_valid('torch.nn.functional.avg_pool3d', generated_inputs['torch.nn.functional.avg_pool3d_4'], lib="torch", suffix=4)
