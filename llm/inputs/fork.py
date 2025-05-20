
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def fork_inputs():
    list_of_inputs = []

    # Example 1: Forking a simple function with a tensor argument
    def foo1(a):
        return a + 1

    input1 = torch.randn(2, 3).numpy()
    input_dict1 = {
        "func": foo1,
        "args": [input1],
        "kwargs": {}
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    return list_of_inputs

generated_inputs = fork_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('fork', generated_inputs)
