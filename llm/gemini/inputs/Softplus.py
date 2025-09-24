
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def softplus_inputs():
    generated_inputs = []

    # Case 1: Default beta and threshold, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Custom beta, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"beta": 2, "input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Custom threshold, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"threshold": 10, "input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Custom beta and threshold, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"beta": 0.5, "threshold": 30, "input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Beta is zero, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"beta": 0, "input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))
    
    # Case 6: Negative Beta, with input
    input_val = np.random.randn(2, 3).astype(np.float32)
    input_dict = {"beta": -1, "input": input_val}
    generated_inputs.append(copy.deepcopy(input_dict))

    return generated_inputs

generated_inputs = softplus_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('Softplus', generated_inputs)
