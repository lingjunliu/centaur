
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def vecdot_inputs():
    list_of_inputs = []

    # Case 1: Basic float tensors, default dim
    x = np.random.randn(3, 2).astype(np.float32)
    y = np.random.randn(3, 2).astype(np.float32)
    input_dict = {"x": x, "y": y, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 2: Complex tensors, default dim
    x = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    y = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    input_dict = {"x": x, "y": y, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 3: Broadcasting, specified dim
    x = np.random.randn(2, 1, 3).astype(np.float64)
    y = np.random.randn(2, 4, 3).astype(np.float64)
    input_dict = {"x": x, "y": y, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 4: Different shapes, negative dim
    x = np.random.randn(5, 4, 2).astype(np.float32)
    y = np.random.randn(5, 4, 2).astype(np.float32)
    input_dict = {"x": x, "y": y, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Case 5: Scalar input, dimension 0
    x = np.random.randn(3).astype(np.float32)
    y = np.random.randn(3).astype(np.float32)
    input_dict = {"x": x, "y": y, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = vecdot_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('vecdot', generated_inputs)
