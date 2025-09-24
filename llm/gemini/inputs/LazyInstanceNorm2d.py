
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def lazy_instance_norm2d_inputs():
    generated_inputs = []

    # Input 1: Basic example with default parameters
    input1 = torch.randn(2, 3, 32, 32).numpy()
    input_dict1 = {
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': True,
        'track_running_stats': True,
        'dtype': None,
        'input': input1
    }
    generated_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Different eps and momentum
    input2 = torch.randn(1, 5, 16, 16).numpy()
    input_dict2 = {
        'eps': 1e-03,
        'momentum': 0.2,
        'affine': True,
        'track_running_stats': True,
        'dtype': None,
        'input': input2
    }
    generated_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: affine=False, track_running_stats=False
    input3 = torch.randn(4, 7, 64, 64).numpy()
    input_dict3 = {
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': False,
        'track_running_stats': False,
        'dtype': None,
        'input': input3
    }
    generated_inputs.append(copy.deepcopy(input_dict3))

    # Input 5: (C, H, W) input - REMOVED: This input caused an error because the driver expected an NCHW input and not CHW
    # input5 = torch.randn(1, 24, 24).numpy()
    # input_dict5 = {
    #     'eps': 1e-05,
    #     'momentum': 0.1,
    #     'affine': True,
    #     'track_running_stats': True,
    #     'dtype': None,
    #     'input': input5
    # }
    # generated_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Small input size
    input6 = torch.randn(1, 3, 2, 2).numpy()
    input_dict6 = {
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': True,
        'track_running_stats': True,
        'dtype': None,
        'input': input6
    }
    generated_inputs.append(copy.deepcopy(input_dict6))

    return generated_inputs

generated_inputs = lazy_instance_norm2d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('LazyInstanceNorm2d', generated_inputs)
