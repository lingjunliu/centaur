
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def unique_inputs():
    list_of_inputs = []

    # Test case 1: 1D integer tensor
    input1 = np.array([1, 3, 2, 3, 1], dtype=np.int64)
    input_dict1 = {
        "input": input1,
        "sorted": True,
        "return_inverse": False,
        "return_counts": False,
        "dim": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))


    return list_of_inputs

generated_inputs = unique_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('unique', generated_inputs)
