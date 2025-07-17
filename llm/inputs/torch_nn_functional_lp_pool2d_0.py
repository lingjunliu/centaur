
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch
import numpy as np
import copy
import torch.nn.functional as F


def lp_pool2d_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    input1 = np.random.rand(1, 1, 10, 10).astype(np.float32)
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
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different kernel size and stride
    input2 = np.random.rand(1, 3, 20, 20).astype(np.float32)
    kernel_size2 = (3, 3)
    stride2 = (1, 1)
    padding2 = (1, 1)
    ceil_mode2 = False
    p2 = 2

    input_dict2 = {
        "input": input2,
        "kernel_size": kernel_size2,
        "stride": stride2,
        "padding": padding2,
        "ceil_mode": ceil_mode2,
        "p": p2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Ceil mode enabled
    input3 = np.random.rand(1, 1, 11, 11).astype(np.float32)
    kernel_size3 = (2, 2)
    stride3 = (2, 2)
    padding3 = (0, 0)
    ceil_mode3 = True
    p3 = 2

    input_dict3 = {
        "input": input3,
        "kernel_size": kernel_size3,
        "stride": stride3,
        "padding": padding3,
        "ceil_mode": ceil_mode3,
        "p": p3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    return list_of_inputs

generated_inputs = {}
generated_inputs["torch.nn.functional.lp_pool2d"] = lp_pool2d_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'torch.nn.functional.lp_pool2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.nn.functional.lp_pool2d'.")

check_valid('torch.nn.functional.lp_pool2d', generated_inputs['torch.nn.functional.lp_pool2d'], lib="torch", suffix=0)
