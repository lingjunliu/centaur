
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy

def lp_pool2d_inputs():
    list_of_inputs = []

    # Input 1
    input1 = torch.randn(1, 3, 32, 32).numpy()
    kernel_size1 = (2, 2)
    stride1 = (2, 2)
    padding1 = (0, 0)
    ceil_mode1 = False
    p1 = 2

    input_dict1 = {
        "input": input1,
        "kernel_size": kernel_size1,
        "stride": stride1,
        "padding": padding1,
        "ceil_mode": ceil_mode1,
        "p": p1
    }
    list_of_inputs.append(input_dict1)

    # Input 2
    input2 = torch.randn(1, 3, 32, 32).numpy()
    kernel_size2 = (3, 3)
    stride2 = (1, 1)
    padding2 = (1, 1)
    ceil_mode2 = True
    p2 = 3

    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "ceil_mode": ceil_mode2,
        "p": p2
    }
    list_of_inputs.append(input_dict2)

    # Input 3
    input3 = torch.randn(2, 3, 16, 16).numpy()
    kernel_size3 = (4, 4)
    stride3 = (4, 4)
    padding3 = (0, 0)
    ceil_mode3 = False
    p3 = 1

    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "ceil_mode": ceil_mode3,
        "p": p3
    }
    list_of_inputs.append(input_dict3)

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.lp_pool2d"] = []
for input_dict in lp_pool2d_inputs():
    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    generated_inputs["torch.nn.functional.lp_pool2d"].append({
        "input": input_tensor,
        "kernel_size": input_dict["kernel_size"],
        "stride": input_dict["stride"],
        "padding": input_dict["padding"],
        "ceil_mode": input_dict["ceil_mode"],
        "p": input_dict["p"]
    })

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.lp_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.lp_pool2d'.")

check_valid('torch.nn.functional.lp_pool2d', generated_inputs['torch.nn.functional.lp_pool2d'], lib="torch", suffix=0)
