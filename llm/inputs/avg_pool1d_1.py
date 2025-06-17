
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch, copy
import numpy as np

def avg_pool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with float tensor
    input1 = torch.randn(1, 3, 10).numpy()
    input_dict1 = {
        "input": input1,
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor
    input2 = torch.randint(0, 10, (1, 2, 15)).numpy()
    input_dict2 = {
        "input": input2,
        "kernel_size": 4,
        "stride": 3,
        "padding": 1,
        "ceil_mode": True,
        "count_include_pad": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values, different shape
    input3 = torch.randn(2, 4, 8).numpy() * -1
    input_dict3 = {
        "input": input3,
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: No stride (stride = kernel_size)
    input4 = torch.randn(1, 1, 20).numpy()
    input_dict4 = {
        "input": input4,
        "kernel_size": 5,
        "stride": 5,
        "padding": 2,
        "ceil_mode": True,
        "count_include_pad": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Larger input
    input5 = torch.randn(4, 8, 30).numpy()
    input_dict5 = {
        "input": input5,
        "kernel_size": 7,
        "stride": 4,
        "padding": 3,
        "ceil_mode": False,
        "count_include_pad": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs


def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.avg_pool1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.avg_pool1d_1'.")

check_valid('torch.nn.functional.avg_pool1d', generated_inputs['torch.nn.functional.avg_pool1d_1'], lib="torch")
