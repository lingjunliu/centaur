
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = torch.randn(1, 1, 10).numpy()
    kernel_size = np.int64(3)
    stride = np.int64(2)
    padding = np.int64(1)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = torch.randn(1, 3, 20).numpy()
    kernel_size = np.int64(5)
    stride = np.int64(3)
    padding = np.int64(2)
    ceil_mode = True
    count_include_pad = False

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = torch.randn(2, 2, 15).numpy()
    kernel_size = np.int64(4)
    stride = np.int64(4)
    padding = np.int64(0)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = torch.randn(1, 4, 30).numpy()
    kernel_size = np.int64(7)
    stride = np.int64(1)
    padding = np.int64(3)
    ceil_mode = True
    count_include_pad = False

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = torch.randn(3, 1, 12).numpy()
    kernel_size = np.int64(2)
    stride = np.int64(1)
    padding = np.int64(1)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_tensor,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.avg_pool1d_1"] = avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool1d_1'.")

check_valid('torch.nn.functional.avg_pool1d', generated_inputs['torch.nn.functional.avg_pool1d_1'], lib="torch", suffix=1)
