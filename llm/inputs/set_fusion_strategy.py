
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def set_fusion_strategy_inputs():
    list_of_inputs = []

    # Strategy 1: Empty list (default behavior)
    input_dict = {
        "strategy": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Strategy 2: Valid fusion strategy (LEVEL0)
    input_dict = {
        "strategy": [0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Strategy 3: Valid fusion strategy (LEVEL1)
    input_dict = {
        "strategy": [1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Strategy 4: Valid fusion strategy (multiple levels - common case)
    input_dict = {
        "strategy": [0, 1]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Strategy 5: Just LEVEL 2
    input_dict = {
        "strategy": [2]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = set_fusion_strategy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('set_fusion_strategy', generated_inputs)
