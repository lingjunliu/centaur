
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def avg_pool3d_inputs():
    list_of_inputs = []

    # Input 1
    input = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4).numpy()
    kernel_size = (2, 2, 2)
    stride = 2
    padding = 0
    ceil_mode = False
    count_include_pad = True
    divisor_override = 8
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
    input = torch.randn(2, 3, 5, 6, 7, dtype=torch.float32).numpy()
    kernel_size = (3, 3, 3)
    stride = 2
    padding = 1
    ceil_mode = False
    count_include_pad = False
    divisor_override = 27
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
    input = torch.linspace(-1, 1, steps=45, dtype=torch.float32).reshape(1, 1, 3, 3, 5).numpy()
    kernel_size = (1, 2, 2)
    stride = 1
    padding = 0
    ceil_mode = True
    count_include_pad = True
    divisor_override = 4
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
    input = torch.randn(1, 2, 1, 2, 2, dtype=torch.float32).numpy()
    kernel_size = (1, 2, 2)
    stride = 2
    padding = 0
    ceil_mode = False
    count_include_pad = False
    divisor_override = 4
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
    input = torch.randn(1, 2, 7, 7, 7, dtype=torch.float64).numpy()
    kernel_size = (3, 2, 2)
    stride = 3
    padding = 0
    ceil_mode = True
    count_include_pad = True
    divisor_override = 12
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
    input = torch.ones(1, 1, 4, 4, 4, dtype=torch.float32).mul(-2.0).numpy()
    kernel_size = (3, 3, 3)
    stride = 1
    padding = 1
    ceil_mode = False
    count_include_pad = True
    divisor_override = 27
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

    # Input 7 (padding fixed to 0 to satisfy kernel=1 dims)
    input = torch.randn(4, 8, 8, 5, 3, dtype=torch.float32).numpy()
    kernel_size = (2, 2, 1)
    stride = 2
    padding = 0
    ceil_mode = False
    count_include_pad = True
    divisor_override = 4
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
    input = torch.arange(2*1*10*9*12, dtype=torch.float32).reshape(2, 1, 10, 9, 12).mul(0.01).numpy()
    kernel_size = (2, 3, 4)
    stride = 3
    padding = 0
    ceil_mode = True
    count_include_pad = False
    divisor_override = 24
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

    # Input 9 (padding fixed to 0 to avoid kernel=1 with pad=1)
    input = torch.randn(1, 3, 2, 3, 4, dtype=torch.float32).numpy()
    kernel_size = (2, 1, 2)
    stride = 1
    padding = 0
    ceil_mode = True
    count_include_pad = True
    divisor_override = 4
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
    input = torch.randn(3, 2, 6, 6, 6, dtype=torch.float64).numpy()
    kernel_size = (3, 3, 3)
    stride = 1
    padding = 0
    ceil_mode = False
    count_include_pad = False
    divisor_override = 27
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

generated_inputs["torch.nn.functional.avg_pool3d_5"] = avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.avg_pool3d_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool3d_5'.")


check_valid('torch.nn.functional.avg_pool3d', generated_inputs['torch.nn.functional.avg_pool3d_5'], lib="torch", suffix=5)
