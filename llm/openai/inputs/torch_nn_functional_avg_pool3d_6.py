
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy, numpy as np

def avg_pool3d_inputs():
    list_of_inputs = []

    input = torch.arange(64, dtype=torch.float32).reshape(1, 1, 4, 4, 4).numpy()
    kernel_size = 2
    stride = [2, 2, 2]
    padding = [0, 0, 0]
    ceil_mode = False
    count_include_pad = True
    divisor_override = 8
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = torch.randn(2, 3, 5, 6, 7, dtype=torch.float32).numpy()
    kernel_size = 3
    stride = [1, 2, 3]
    padding = [1, 1, 1]
    ceil_mode = False
    count_include_pad = False
    divisor_override = 3
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = torch.linspace(-10, 10, steps=1*2*3*4*5, dtype=torch.float64).reshape(1, 2, 3, 4, 5).numpy()
    kernel_size = 1
    stride = [1, 1, 1]
    padding = [0, 0, 0]
    ceil_mode = True
    count_include_pad = True
    divisor_override = 1
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = torch.randn(3, 1, 8, 3, 2, dtype=torch.float32).numpy()
    kernel_size = 2
    stride = [2, 1, 2]
    padding = [0, 1, 0]
    ceil_mode = False
    count_include_pad = True
    divisor_override = 4
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = (torch.ones(1, 1, 7, 7, 7, dtype=torch.float32) * 5).numpy()
    kernel_size = 4
    stride = [3, 3, 3]
    padding = [1, 1, 1]
    ceil_mode = True
    count_include_pad = False
    divisor_override = 4
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = (torch.arange(5*2*9*9*1, dtype=torch.float32) - 100).reshape(5, 2, 9, 9, 1).numpy()
    kernel_size = 1
    stride = [2, 3, 1]
    padding = [0, 0, 0]
    ceil_mode = False
    count_include_pad = True
    divisor_override = 1
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = torch.randn(1, 4, 2, 2, 10, dtype=torch.float32).numpy()
    kernel_size = 2
    stride = [2, 2, 5]
    padding = [0, 0, 0]
    ceil_mode = False
    count_include_pad = False
    divisor_override = 2
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = torch.randn(2, 2, 6, 6, 6, dtype=torch.float32).numpy()
    kernel_size = 3
    stride = [2, 2, 2]
    padding = [1, 0, 1]
    ceil_mode = True
    count_include_pad = True
    divisor_override = 3
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = torch.randn(1, 1, 3, 8, 8, dtype=torch.float32).numpy()
    kernel_size = 2
    stride = [1, 2, 2]
    padding = [0, 0, 0]
    ceil_mode = True
    count_include_pad = False
    divisor_override = 2
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = torch.randn(2, 1, 2, 5, 5, dtype=torch.float32).numpy()
    kernel_size = 2
    stride = [2, 2, 2]
    padding = [1, 0, 0]
    ceil_mode = True
    count_include_pad = True
    divisor_override = 4
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = torch.randn(1, 2, 10, 5, 5, dtype=torch.float64).numpy()
    kernel_size = 3
    stride = [3, 1, 1]
    padding = [0, 1, 1]
    ceil_mode = False
    count_include_pad = False
    divisor_override = 3
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    input = torch.tensor(np.random.uniform(-1, 1, (4, 3, 4, 4, 4)), dtype=torch.float32).numpy()
    kernel_size = 2
    stride = [1, 1, 2]
    padding = [1, 0, 1]
    ceil_mode = True
    count_include_pad = True
    divisor_override = 2
    list_of_inputs.append(copy.deepcopy({
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override
    }))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool3d_6"] = avg_pool3d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.avg_pool3d_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool3d_6'.")


check_valid('torch.nn.functional.avg_pool3d', generated_inputs['torch.nn.functional.avg_pool3d_6'], lib="torch", suffix=6)
