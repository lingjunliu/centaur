
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def vector_norm_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D tensor with default parameters
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {
        "input": input1,
        "ord": None,
        "dim": None,
        "keepdim": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D tensor with specified ord and dim
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict2 = {
        "input": input2,
        "ord": 1.0,
        "dim": 1,
        "keepdim": True,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D tensor with negative values and specified dtype
    input3 = np.array([[[1.0, -2.0], [3.0, -4.0]], [[-5.0, 6.0], [-7.0, 8.0]]], dtype=np.float64)
    input_dict3 = {
        "input": input3,
        "ord": 2.0,
        "dim": (1, 2),
        "keepdim": False,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 5: Matrix norm - REMOVED DUE TO ERROR - string not supported
    # Input 6: Negative Ord
    input6 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict6 = {
        "input": input6,
        "ord": -1.0,
        "dim": None,
        "keepdim": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: 4D tensor
    input7 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_dict7 = {
        "input": input7,
        "ord": 2.0,
        "dim": (1,2),
        "keepdim": True,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input8 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict8 = {
        "input": input8,
        "ord": np.inf,
        "dim": None,
        "keepdim": False,
        "dtype": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    return list_of_inputs

generated_inputs = vector_norm_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('vector_norm', generated_inputs)
