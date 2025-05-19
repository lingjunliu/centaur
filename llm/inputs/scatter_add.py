
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def scatter_add_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float tensor
    input_dict = {
        "input": np.zeros((5, 3), dtype=np.float32),
        "dim": 0,
        "index": np.array([[0, 1, 2], [0, 1, 2], [0, 2, 2]], dtype=np.int64),
        "src": np.ones((3, 3), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer tensor, dim=1
    input_dict = {
        "input": np.zeros((2, 4), dtype=np.int64),
        "dim": 1,
        "index": np.array([[0, 2], [1, 3]], dtype=np.int64),
        "src": np.ones((2, 2), dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor, negative values in src
    input_dict = {
        "input": np.zeros((3, 2, 2), dtype=np.float32),
        "dim": 0,
        "index": np.array([[[0, 0], [1, 1], [2, 0]], [[0, 1], [1, 0], [2, 1]]], dtype=np.int64),
        "src": np.array([[[1, -1], [1, -1]], [[-1, 1], [-1, 1]], [[1, 1], [-1, -1]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 4: Different shape for src, broadcasting
    input_dict = {
        "input": np.zeros((5, 3), dtype=np.float32),
        "dim": 0,
        "index": np.array([[0, 1, 2]], dtype=np.int64),
        "src": np.ones((1, 3), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Another example with int64, limiting indices for the 3D input case
    input_dict = {
        "input": np.zeros((3, 2, 2), dtype=np.int64),
        "dim": 0,
        "index": np.array([[[0, 0], [1, 1], [2, 0]], [[0, 1], [1, 0], [2, 1]]], dtype=np.int64),
        "src": np.ones((3, 2, 2), dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = scatter_add_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('scatter_add', generated_inputs)
