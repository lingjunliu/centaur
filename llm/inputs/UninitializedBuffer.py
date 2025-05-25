
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np
import torch.nn as nn

def uninitialized_buffer_inputs():
    list_of_inputs = []

    # Example 1: Basic float tensor
    input_dict = {
        'size': (2, 3),
        'dtype': torch.float32,
        'layout': 'strided',
        'requires_grad': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = uninitialized_buffer_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('UninitializedBuffer', generated_inputs)
