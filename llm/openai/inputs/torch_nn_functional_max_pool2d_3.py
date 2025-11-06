
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def max_pool2d_inputs():
    list_of_inputs = []

    input = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    kernel_size = 2
    stride = (2, 2)
    padding = (0, 0)
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(2, 3, 7, 5).astype(np.float64)
    kernel_size = 3
    stride = (1, 1)
    padding = (1, 1)
    dilation = 1
    return_indices = True
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(1, 2, 5, 5).astype(np.float32)
    kernel_size = 3
    stride = (2, 2)
    padding = (1, 1)
    dilation = 1
    return_indices = False
    ceil_mode = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = -np.abs(np.random.randn(3, 1, 8, 6).astype(np.float32))
    kernel_size = 2
    stride = (2, 1)
    padding = (0, 0)
    dilation = 1
    return_indices = True
    ceil_mode = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.linspace(-1.0, 1.0, num=1*3*10*10, dtype=np.float32).reshape(1, 3, 10, 10)
    kernel_size = 4
    stride = (4, 4)
    padding = (0, 0)
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(1, 1, 9, 7).astype(np.float64)
    kernel_size = 3
    stride = (2, 3)
    padding = (1, 0)
    dilation = 1
    return_indices = True
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(2, 5, 3, 3).astype(np.float32)
    kernel_size = 1
    stride = (1, 1)
    padding = (0, 0)
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(1, 2, 6, 9).astype(np.float64)
    kernel_size = 5
    stride = (3, 4)
    padding = (2, 2)
    dilation = 1
    return_indices = True
    ceil_mode = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(4, 3, 15, 15).astype(np.float64)
    kernel_size = 5
    stride = (5, 5)
    padding = (0, 0)
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = -np.abs(np.random.randn(2, 4, 8, 8).astype(np.float32))
    kernel_size = 3
    stride = (1, 2)
    padding = (0, 1)
    dilation = 1
    return_indices = True
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(1, 1, 10, 6).astype(np.float64)
    kernel_size = 3
    stride = (2, 1)
    padding = (1, 0)
    dilation = 1
    return_indices = False
    ceil_mode = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(1, 1, 5, 5).astype(np.float32)
    kernel_size = 2
    stride = (2, 2)
    padding = (1, 1)
    dilation = 1
    return_indices = True
    ceil_mode = True
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input = np.random.randn(2, 2, 12, 7).astype(np.float32)
    kernel_size = 3
    stride = (2, 2)
    padding = (0, 1)
    dilation = 1
    return_indices = False
    ceil_mode = False
    input_dict = {
        "input": input,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "return_indices": return_indices,
        "ceil_mode": ceil_mode
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.max_pool2d_3"] = max_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.nn.functional.max_pool2d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.max_pool2d_3'.")


check_valid('torch.nn.functional.max_pool2d', generated_inputs['torch.nn.functional.max_pool2d_3'], lib="torch", suffix=3)
