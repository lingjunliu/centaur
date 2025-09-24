
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.random.rand(2, 3, 10, 10, 10).astype(np.float32)
    kernel_size = [3, 3, 3]
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    ceil_mode = False
    count_include_pad = True
    divisor_override = 1

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    kernel_size = [2, 2, 2]
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    ceil_mode = True
    count_include_pad = False
    divisor_override = 1

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.random.rand(4, 2, 8, 8, 8).astype(np.float32)
    kernel_size = [4, 4, 4]
    stride = (3, 3, 3)
    padding = (2, 2, 2)
    ceil_mode = False
    count_include_pad = False
    divisor_override = 1

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.random.rand(1, 3, 12, 12, 12).astype(np.float32)
    kernel_size = [5, 5, 5]
    stride = (4, 4, 4)
    padding = (0, 0, 0)
    ceil_mode = True
    count_include_pad = True
    divisor_override = 1

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.random.rand(2, 1, 7, 7, 7).astype(np.float32)
    kernel_size = [3, 3, 3]
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    ceil_mode = False
    count_include_pad = True
    divisor_override = 1

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.random.rand(3, 4, 9, 9, 9).astype(np.float32)
    kernel_size = [2, 2, 2]
    stride = (1, 2, 1)
    padding = (0, 1, 0)
    ceil_mode = True
    count_include_pad = False
    divisor_override = 1

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7
    input_tensor = np.random.rand(1, 2, 6, 6, 6).astype(np.float32)
    kernel_size = [3, 2, 1]
    stride = (2, 1, 1)
    padding = (1, 0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = 1

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.random.rand(2, 3, 11, 11, 11).astype(np.float32)
    kernel_size = [1, 2, 3]
    stride = (1, 1, 2)
    padding = (0, 1, 1) #modified padding
    ceil_mode = True
    count_include_pad = False
    divisor_override = 1

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.random.rand(1, 1, 4, 4, 4).astype(np.float32)
    kernel_size = [2, 2, 2]
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    ceil_mode = False
    count_include_pad = True
    divisor_override = 1

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.random.rand(4, 2, 13, 13, 13).astype(np.float32)
    kernel_size = [3, 3, 3]
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    ceil_mode = True
    count_include_pad = False
    divisor_override = 1

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.avg_pool3d_9"] = avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool3d_9' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool3d_9'.")

check_valid('torch.nn.functional.avg_pool3d', generated_inputs['torch.nn.functional.avg_pool3d_9'], lib="torch", suffix=9)
