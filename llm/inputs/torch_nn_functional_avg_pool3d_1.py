
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input = np.random.rand(1, 1, 10, 10, 10).astype(np.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    ceil_mode = False
    count_include_pad = True
    divisor_override = 1

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input = np.random.rand(2, 3, 12, 12, 12).astype(np.float32)
    kernel_size = 2
    stride = 2
    padding = 0
    ceil_mode = True
    count_include_pad = False
    divisor_override = 1

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input = np.random.rand(1, 1, 5, 5, 5).astype(np.float32)
    kernel_size = 1
    stride = 1
    padding = 0
    ceil_mode = False
    count_include_pad = True
    divisor_override = 2

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4
    input = np.random.rand(4, 2, 8, 8, 8).astype(np.float32)
    kernel_size = 4
    stride = 4
    padding = 2
    ceil_mode = True
    count_include_pad = False
    divisor_override = 3

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input = np.random.rand(1, 1, 7, 7, 7).astype(np.float32)
    kernel_size = 3
    stride = 1
    padding = 1
    ceil_mode = False
    count_include_pad = True
    divisor_override = 1

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6
    input = np.random.rand(1, 1, 10, 10, 10).astype(np.float32)
    kernel_size = 3
    stride = 3
    padding = 0
    ceil_mode = True
    count_include_pad = False
    divisor_override = 1

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input = np.random.rand(2, 3, 10, 10, 10).astype(np.float32)
    kernel_size = 2
    stride = 1
    padding = 0
    ceil_mode = False
    count_include_pad = False
    divisor_override = 1

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input = np.random.rand(1, 1, 6, 6, 6).astype(np.float32)
    kernel_size = 2
    stride = 2
    padding = 1
    ceil_mode = True
    count_include_pad = True
    divisor_override = 1

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    input = np.random.rand(1, 1, 4, 4, 4).astype(np.float32)
    kernel_size = 1
    stride = 1
    padding = 0
    ceil_mode = False
    count_include_pad = False
    divisor_override = 1

    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input = np.random.rand(3, 2, 11, 11, 11).astype(np.float32)
    kernel_size = 3
    stride = 2
    padding = 1
    ceil_mode = True
    count_include_pad = True
    divisor_override = 1

    input_dict = {
        "input": input,
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
generated_inputs["torch.nn.functional.avg_pool3d_1"] = avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool3d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool3d_1'.")

check_valid('torch.nn.functional.avg_pool3d', generated_inputs['torch.nn.functional.avg_pool3d_1'], lib="torch", suffix=1)
