
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def HuberLoss_inputs():
    list_of_inputs = []

    # Case 1: Default values
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.5, 2.5, 3.5]),
        "reduction": 'mean',
        "delta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Different reduction
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.5, 2.5, 3.5]),
        "reduction": 'sum',
        "delta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: No reduction
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.5, 2.5, 3.5]),
        "reduction": 'none',
        "delta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different delta
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.5, 2.5, 3.5]),
        "reduction": 'mean',
        "delta": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Larger delta
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.5, 2.5, 3.5]),
        "reduction": 'mean',
        "delta": 2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = HuberLoss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('HuberLoss', generated_inputs)
