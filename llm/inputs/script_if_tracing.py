
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def script_if_tracing_inputs():
    list_of_inputs = []

    # Example 1: Condition True, simple lists
    input_dict = {
        "condition": True,
        "fn": [1, 2, 3],
        "alternative_fn": [4, 5, 6]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Condition False, lists of different types
    input_dict = {
        "condition": False,
        "fn": [1.0, 2.5, 3.7],
        "alternative_fn": ["a", "b", "c"]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 3: Condition True, empty lists
    input_dict = {
        "condition": True,
        "fn": [],
        "alternative_fn": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 4: Condition False, simple numerical lists
    input_dict = {
        "condition": False,
        "fn": [1, 2],
        "alternative_fn": [3, 4]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 5: Condition True, lists with mixed types
    input_dict = {
        "condition": True,
        "fn": [1, "hello", 3.14],
        "alternative_fn": [False, 42, "world"]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = script_if_tracing_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('script_if_tracing', generated_inputs)
