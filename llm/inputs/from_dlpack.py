
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def from_dlpack_inputs():
    list_of_inputs = []

    # Input 1: Float tensor, 1D
    arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"dlpack": arr1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensor, 2D, including negative values
    arr2 = np.array([[-1, 0, 1], [2, -3, 4]], dtype=np.int32)
    input_dict2 = {"dlpack": arr2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Float tensor, 3D
    arr3 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict3 = {"dlpack": arr3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Boolean tensor, 2D
    arr4 = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict4 = {"dlpack": arr4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Complex tensor, 1D
    arr5 = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    input_dict5 = {"dlpack": arr5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = from_dlpack_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('from_dlpack', generated_inputs)
