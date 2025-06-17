
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input_data = torch.randn(1, 3, 10).numpy()
    kernel_size = (2,)
    stride = (2,)
    padding = (0,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_data,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different stride and kernel size
    input_data = torch.randn(1, 5, 15).numpy()
    kernel_size = (3,)
    stride = (1,)
    padding = (1,)
    ceil_mode = False
    count_include_pad = True

    input_dict = {
        "input": input_data,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Ceil mode enabled
    input_data = torch.randn(1, 2, 7).numpy()
    kernel_size = (2,)
    stride = (3,)
    padding = (0,)
    ceil_mode = True
    count_include_pad = True

    input_dict = {
        "input": input_data,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }

    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs["torch.nn.functional.avg_pool1d_2"] = avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool1d_2'.")

check_valid('torch.nn.functional.avg_pool1d', generated_inputs['torch.nn.functional.avg_pool1d_2'], lib="torch")
