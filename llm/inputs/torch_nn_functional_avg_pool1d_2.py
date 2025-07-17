
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = (3,)
    stride = (1,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = (2,)
    stride = (2,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = (2,)
    stride = (2,)
    padding = (1,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = (2,)
    stride = (2,)
    padding = (1,)
    ceil_mode = True
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = (2,)
    stride = (2,)
    padding = (1,)
    ceil_mode = False
    count_include_pad = False

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = (5,)
    stride = (1,)
    padding = (2,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    kernel_size = (5,)
    stride = (1,)
    padding = (2,)
    ceil_mode = True
    count_include_pad = False

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]], dtype=np.float32)
    kernel_size = (3,)
    stride = (1,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]], dtype=np.float32)
    kernel_size = (2,)
    stride = (2,)
    padding = (1,)
    ceil_mode = True
    count_include_pad = False

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]], [[6.0, 7.0, 8.0, 9.0, 10.0]]], dtype=np.float32)
    kernel_size = (2,)
    stride = (2,)
    padding = (1,)
    ceil_mode = True
    count_include_pad = False

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]], dtype=np.float32)
    kernel_size = (4,)
    stride = (1,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]], dtype=np.float32)
    kernel_size = (4,)
    stride = (1,)
    padding = (2,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Negative values in input
    input_tensor = np.array([[[ -1.0, -2.0, 3.0, -4.0, 5.0]]], dtype=np.float32)
    kernel_size = (3,)
    stride = (1,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: Larger Kernel Size
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]], dtype=np.float32)
    kernel_size = (6,)
    stride = (2,)
    padding = (1,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 15: Different dtypes
    input_tensor = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float64)
    kernel_size = (3,)
    stride = (1,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool1d_2"] = avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool1d_2'.")

check_valid('torch.nn.functional.avg_pool1d', generated_inputs['torch.nn.functional.avg_pool1d_2'], lib="torch", suffix=2)
