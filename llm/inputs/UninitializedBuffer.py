
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def uninitializedbuffer_inputs():
    list_of_inputs = []

    # Example 1: Basic float32 tensor
    input_dict = {
        "size": (2, 3, 4),
        "dtype": torch.float32,
        "layout": 'strided',
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Int64 tensor
    input_dict = {
        "size": (5, 5),
        "dtype": torch.int64,
        "layout": 'strided',
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = uninitializedbuffer_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('UninitializedBuffer', generated_inputs)
