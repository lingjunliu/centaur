
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def fractionalmaxpool3d_inputs():
    list_of_inputs = []

    # Case 1: Using output_size
    input_dict = {
        'kernel_size': (3, 3, 3),
        'output_size': (13, 12, 11),
        'output_ratio': None,
        'return_indices': False,
        '_random_samples': torch.rand(1, 1, 1).numpy(),
        'input': torch.randn(20, 16, 50, 32, 16).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Using output_ratio
    input_dict = {
        'kernel_size': (3, 3, 3),
        'output_size': None,
        'output_ratio': (0.5, 0.5, 0.5),
        'return_indices': False,
        '_random_samples': torch.rand(1, 1, 1).numpy(),
        'input': torch.randn(20, 16, 50, 32, 16).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Integer kernel_size and return_indices=True
    input_dict = {
        'kernel_size': 2,
        'output_size': (7, 8, 9),
        'output_ratio': None,
        'return_indices': True,
        '_random_samples': torch.rand(1, 1, 1).numpy(),
        'input': torch.randn(20, 16, 50, 32, 16).numpy()
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = fractionalmaxpool3d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('FractionalMaxPool3d', generated_inputs)
