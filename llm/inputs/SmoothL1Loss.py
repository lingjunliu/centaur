
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def smoothl1loss_inputs():
    list_of_inputs = []

    # Test case 1: Default parameters
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.1, 2.2, 3.3]),
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 2: Different reduction
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.1, 2.2, 3.3]),
        "size_average": None,
        "reduce": None,
        "reduction": 'sum',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Test case 3: No reduction
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.1, 2.2, 3.3]),
        "size_average": None,
        "reduce": None,
        "reduction": 'none',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 4: Different beta
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.1, 2.2, 3.3]),
        "size_average": None,
        "reduce": None,
        "reduction": 'mean',
        "beta": 0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Test case 5: size_average = False
    input_dict = {
        "input": np.array([1.0, 2.0, 3.0]),
        "target": np.array([1.1, 2.2, 3.3]),
        "size_average": False,
        "reduce": None,
        "reduction": 'mean',
        "beta": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = smoothl1loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('SmoothL1Loss', generated_inputs)
