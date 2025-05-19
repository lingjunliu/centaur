
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def nll_loss_inputs():
    list_of_inputs = []

    # Case 1: Basic case with float input and long target
    input_dict = {
        "input": np.array([[-0.5, -0.2, -0.3], [-0.1, -0.8, -0.1]], dtype=np.float64),
        "target": np.array([0, 2], dtype=np.int64),
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2:  Input with different shape and weights provided
    input_dict = {
        "input": np.array([[-1.2, -0.3, -0.5, -0.1]], dtype=np.float64),
        "target": np.array([3], dtype=np.int64),
        "weight": np.array([0.2, 0.3, 0.1, 0.4], dtype=np.float64),
        "ignore_index": -100,
        "reduction": 'sum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Input with ignore_index
    input_dict = {
        "input": np.array([[-0.7, -0.1, -0.2], [-0.4, -0.5, -0.1], [-0.2, -0.6, -0.2]], dtype=np.float64),
        "target": np.array([0, 1, 2], dtype=np.int64),
        "weight": None,
        "ignore_index": 1,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Input with 'none' reduction
    input_dict = {
        "input": np.array([[-0.1, -0.9, -0.0], [-0.3, -0.4, -0.3]], dtype=np.float64),
        "target": np.array([2, 0], dtype=np.int64),
        "weight": None,
        "ignore_index": -100,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Input with a weight and ignore_index specified.
    input_dict = {
        "input": np.array([[-0.6, -0.2, -0.2], [-0.3, -0.3, -0.4]], dtype=np.float64),
        "target": np.array([0, 1], dtype=np.int64),
        "weight": np.array([0.5, 0.3, 0.2], dtype=np.float64),
        "ignore_index": 0,
        "reduction": 'mean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = nll_loss_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('nll_loss', generated_inputs)
