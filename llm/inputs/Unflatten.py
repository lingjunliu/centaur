
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def unflatten_inputs():
    list_of_inputs = []

    # Example 1
    input_tensor = torch.randn(2, 50).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "unflattened_size": (2, 5, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2
    input_tensor = torch.randn(1, 100).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "unflattened_size": (10, 10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3
    input_tensor = torch.randn(2, 25).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 1,
        "unflattened_size": (5, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Example 4: Negative dimension
    input_tensor = torch.randn(2, 50).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": -1,
        "unflattened_size": (5, 10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Unflatten dimension 0
    input_tensor = torch.randn(10).numpy()
    input_dict = {
        "input": input_tensor,
        "dim": 0,
        "unflattened_size": (2, 5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = unflatten_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Unflatten', generated_inputs)
