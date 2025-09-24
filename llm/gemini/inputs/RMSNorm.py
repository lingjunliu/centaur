
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def rmsnorm_inputs():
    list_of_inputs = []

    # Test case 1: Basic case with a single integer normalized_shape
    input_dict = {
        "normalized_shape": [3],
        "eps": 1e-05,
        "elementwise_affine": True,
        "dtype": torch.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = rmsnorm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('RMSNorm', generated_inputs)
