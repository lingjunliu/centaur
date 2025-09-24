
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import numpy as np

def hamming_window_inputs():
    generated_inputs = []

    input_dict = {
        "window_length": 5,
        "periodic": False,
        "alpha": 0.5,
        "beta": 0.5,
        "dtype": torch.float32
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 10,
        "periodic": True,
        "alpha": 0.5,
        "beta": 0.5,
        "dtype": torch.float64
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 7,
        "periodic": False,
        "alpha": 0.0,
        "beta": 1.0,
        "dtype": torch.float32
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 12,
        "periodic": True,
        "alpha": 1.0,
        "beta": 0.0,
        "dtype": torch.float64
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 3,
        "periodic": False,
        "alpha": 0.25,
        "beta": 0.75,
        "dtype": torch.float32
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 8,
        "periodic": True,
        "alpha": 0.75,
        "beta": 0.25,
        "dtype": torch.float64
    }
    generated_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "window_length": 15,
        "periodic": False,
        "alpha": 0.6,
        "beta": 0.4,
        "dtype": torch.float32
    }
    generated_inputs.append(copy.deepcopy(input_dict))
    
    return generated_inputs

generated_inputs = hamming_window_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('hamming_window', generated_inputs)
