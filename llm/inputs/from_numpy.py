
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def from_numpy_inputs():
    list_of_inputs = []

    # Example 1: 1D float array
    arr1 = np.array([1.0, 2.0, 3.0, 4.0])
    input_dict1 = {"a": arr1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Example 2: 2D int array with negative values
    arr2 = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    input_dict2 = {"a": arr2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

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
