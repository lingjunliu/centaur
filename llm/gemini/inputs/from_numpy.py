
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def from_numpy_inputs():
    list_of_inputs = []

    # Example 1: 1D numpy array of integers
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    list_of_inputs.append({"input": a})

    # Example 2: 2D numpy array of floats
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"input": a})

    # Example 3: numpy array with negative values
    a = np.array([-1, 0, 1, -2, 2], dtype=np.int32)
    list_of_inputs.append({"input": a})

    # Example 4: numpy array of booleans
    a = np.array([True, False, True, True, False], dtype=bool)
    list_of_inputs.append({"input": a})
    
    return list_of_inputs

generated_inputs = from_numpy_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('from_numpy', generated_inputs)
