
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def lazy_instance_norm1d_inputs():
    list_of_inputs = []

    # Input 1: Basic case with default parameters
    input_dict = {
        "input": torch.randn(1, 3, 10).numpy(),
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different eps and momentum values
    input_dict = {
        "input": torch.randn(1, 5, 20).numpy(),
        "eps": 1e-04,
        "momentum": 0.2,
        "affine": False,
        "track_running_stats": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using a specific dtype (torch.float64)
    input_dict = {
        "input": torch.randn(1, 2, 15).numpy(),
        "eps": 1e-05,
        "momentum": 0.1,
        "affine": True,
        "track_running_stats": True,
        "dtype": torch.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: No affine transformation, keeping track of running stats
    input_dict = {
        "input": torch.randn(1, 4, 25).numpy(),
        "eps": 1e-06,
        "momentum": 0.05,
        "affine": False,
        "track_running_stats": True,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Affine transformation, no tracking of running stats
    input_dict = {
        "input": torch.randn(1, 6, 30).numpy(),
        "eps": 1e-07,
        "momentum": 0.15,
        "affine": True,
        "track_running_stats": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = lazy_instance_norm1d_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('LazyInstanceNorm1d', generated_inputs)
