
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def adaptive_avg_pool3d_inputs():
    list_of_inputs = []

    # Test case 1: Tuple (D, H, W) with integers
    input_dict = {
        "output_size": (5, 7, 9),
        "input": np.random.randn(1, 64, 8, 9, 10).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Single integer (cube)
    input_dict = {
        "output_size": 7,
        "input": np.random.randn(1, 64, 10, 9, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 3: Tuple with None values
    input_dict = {
        "output_size": (7, None, None),
        "input": np.random.randn(1, 64, 10, 9, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Tuple with mixed int and None
    input_dict = {
        "output_size": (None, 9, None),
        "input": np.random.randn(1, 64, 10, 9, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 5: All None values (should be same as input)
    input_dict = {
        "output_size": (None, None, None),
        "input": np.random.randn(1, 64, 10, 9, 8).astype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = adaptive_avg_pool3d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('AdaptiveAvgPool3d', generated_inputs)
