
from utils.new_api_utils import run_api

generated_inputs = dict()

import torch
import copy
import numpy as np

def LPPool1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    input_dict = {
        "norm_type": 2,
        "kernel_size": 3,
        "stride": 2,
        "ceil_mode": False,
        "input": torch.randn(20, 16, 50).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different norm_type, no stride
    input_dict = {
        "norm_type": 1,
        "kernel_size": 5,
        "stride": None,
        "ceil_mode": True,
        "input": torch.randn(5, 8, 100).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different input shape (C, Lin)
    input_dict = {
        "norm_type": 3,
        "kernel_size": 4,
        "stride": 3,
        "ceil_mode": False,
        "input": torch.randn(16, 50).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different input with negative values
    input_dict = {
        "norm_type": 2,
        "kernel_size": 3,
        "stride": 2,
        "ceil_mode": True,
        "input": torch.randn(2, 4, 20).numpy() * -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: kernel_size = 1
    input_dict = {
        "norm_type": 2,
        "kernel_size": 1,
        "stride": 1,
        "ceil_mode": False,
        "input": torch.randn(10, 5, 30).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: stride greater than kernel size
    input_dict = {
        "norm_type": 2,
        "kernel_size": 2,
        "stride": 3,
        "ceil_mode": False,
        "input": torch.randn(5, 3, 15).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.LPPool1d"] = LPPool1d_inputs()

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

check_valid('torch.nn.LPPool1d', generated_inputs['torch.nn.LPPool1d'], lib="torch")
