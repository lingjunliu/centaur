
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def QFunctional_inputs():
    list_of_inputs = []

    # Example 1: Basic addition with scale and zero_point
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    scale = 0.5
    zero_point = 0

    input_dict = {
        "x": x,
        "y": y,
        "scale": scale,
        "zero_point": zero_point
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Addition with different data types and shapes
    x = np.array([[1, 2], [3, 4]], dtype=np.int8)
    y = np.array([[5, 6], [7, 8]], dtype=np.int8)
    scale = 0.25
    zero_point = 10

    input_dict = {
        "x": x,
        "y": y,
        "scale": scale,
        "zero_point": zero_point
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    scale = 0.5
    zero_point = 0

    input_dict = {
        "x": x,
        "y": y,
        "scale": scale,
        "zero_point": zero_point
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Example 2: Addition with different data types and shapes
    x = np.array([[1, 2], [3, 4]], dtype=np.int8)
    y = np.array([[5, 6], [7, 8]], dtype=np.int8)
    scale = 0.25
    zero_point = 10

    input_dict = {
        "x": x,
        "y": y,
        "scale": scale,
        "zero_point": zero_point
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = QFunctional_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('QFunctional', generated_inputs)
