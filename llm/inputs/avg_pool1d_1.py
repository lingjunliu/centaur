
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    input1 = torch.randn(1, 3, 10).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different kernel size and stride
    input2 = torch.randn(1, 3, 10).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: ceil_mode=True
    input3 = torch.randn(1, 3, 10).numpy()
    input_dict3 = {
        "input": input3,
        "kernel_size": 2,
        "stride": 2,
        "padding": 0,
        "ceil_mode": True,
        "count_include_pad": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Input 4: count_include_pad=False
    input4 = torch.randn(1, 3, 10).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": 2,
        "stride": 2,
        "padding": 1,
        "ceil_mode": False,
        "count_include_pad": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.avg_pool1d_1"] = avg_pool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool1d_1'.")

check_valid('torch.nn.functional.avg_pool1d', generated_inputs['torch.nn.functional.avg_pool1d_1'], lib="torch")
