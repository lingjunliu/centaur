
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import torch.nn as nn
import numpy as np

def remove_weight_norm_inputs():
    list_of_inputs = []

    # Example 1: Linear layer
    linear_module = nn.Linear(10, 20)
    nn.utils.weight_norm(linear_module, name='weight')
    input_dict = {
        "module": linear_module,
        "name": 'weight'
    }
    list_of_inputs.append(input_dict)

    # Example 2: Conv1d layer
    conv1d_module = nn.Conv1d(3, 16, 5)
    nn.utils.weight_norm(conv1d_module, name='weight')
    input_dict = {
        "module": conv1d_module,
        "name": 'weight'
    }
    list_of_inputs.append(input_dict)

    # Example 3: Conv2d layer
    conv2d_module = nn.Conv2d(3, 16, (3, 5))
    nn.utils.weight_norm(conv2d_module, name='weight')
    input_dict = {
        "module": conv2d_module,
        "name": 'weight'
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = remove_weight_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('remove_weight_norm', generated_inputs)
