
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def max_pool3d_inputs_set():
    list_of_inputs = []

    # Input 1
    input = torch.randn(2, 3, 8, 9, 10, dtype=torch.float32).numpy()
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    dilation = (1, 1, 1)
    return_indices = False
    ceil_mode = False
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 2 (4D input)
    input = torch.arange(3*5*6*7, dtype=torch.float32).reshape(3, 5, 6, 7).numpy()
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    return_indices = True
    ceil_mode = False
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 3 (negatives, mixed kernel)
    input = torch.linspace(-2.0, 2.0, steps=64, dtype=torch.float32).reshape(1, 1, 4, 4, 4).numpy()
    kernel_size = (4, 2, 3)
    stride = (4, 2, 3)
    padding = (0, 0, 0)
    dilation = (1, 1, 1)
    return_indices = False
    ceil_mode = False
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 4 (float64, dilation > 1, adjusted padding)
    input = torch.randn(1, 2, 10, 10, 10, dtype=torch.float64).numpy()
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    dilation = (2, 2, 2)
    return_indices = True
    ceil_mode = False
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 5 (asymmetric stride)
    input = torch.randn(4, 2, 7, 8, 9, dtype=torch.float32).numpy()
    kernel_size = (2, 3, 4)
    stride = (1, 2, 3)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    return_indices = False
    ceil_mode = False
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 6 (ceil_mode True)
    input = torch.randn(1, 3, 6, 6, 6, dtype=torch.float32).numpy()
    kernel_size = (2, 2, 2)
    stride = (4, 3, 2)
    padding = (1, 0, 1)
    dilation = (1, 1, 1)
    return_indices = True
    ceil_mode = True
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 7 (monotonic increasing values)
    input = torch.arange(2*1*5*5*5, dtype=torch.float32).reshape(2, 1, 5, 5, 5).numpy()
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    padding = (0, 0, 0)
    dilation = (1, 1, 1)
    return_indices = True
    ceil_mode = True
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 8 (dilation mix, adjusted padding)
    input = torch.randn(1, 1, 9, 10, 11, dtype=torch.float32).numpy()
    kernel_size = (3, 2, 2)
    stride = (2, 3, 2)
    padding = (1, 0, 1)
    dilation = (2, 1, 2)
    return_indices = False
    ceil_mode = False
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 9 (4D input, padding asymmetric)
    input = torch.randn(2, 6, 7, 8, dtype=torch.float32).numpy()
    kernel_size = (2, 2, 2)
    stride = (2, 2, 2)
    padding = (1, 0, 1)
    dilation = (1, 1, 1)
    return_indices = True
    ceil_mode = False
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 10 (kernel 1 with dilation > 1)
    input = torch.randn(5, 4, 3, 3, 3, dtype=torch.float32).numpy()
    kernel_size = (1, 1, 1)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    dilation = (3, 1, 2)
    return_indices = False
    ceil_mode = False
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 11 (larger kernel and stride)
    input = torch.randn(1, 2, 12, 7, 5, dtype=torch.float32).numpy()
    kernel_size = (5, 3, 2)
    stride = (2, 2, 1)
    padding = (2, 1, 0)
    dilation = (1, 2, 1)
    return_indices = True
    ceil_mode = False
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    # Input 12 (ceil with dilation)
    input = torch.randn(1, 3, 9, 9, 9, dtype=torch.float32).numpy()
    kernel_size = (3, 3, 3)
    stride = (4, 4, 4)
    padding = (1, 1, 1)
    dilation = (2, 2, 2)
    return_indices = False
    ceil_mode = True
    list_of_inputs.append(copy.deepcopy({
        "input": input, "kernel_size": kernel_size, "stride": stride,
        "padding": padding, "dilation": dilation,
        "return_indices": return_indices, "ceil_mode": ceil_mode
    }))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool3d_2"] = max_pool3d_inputs_set()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool3d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool3d_2'.")


check_valid('torch.nn.functional.max_pool3d', generated_inputs['torch.nn.functional.max_pool3d_2'], lib="torch", suffix=2)
