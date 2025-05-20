
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def aminmax_inputs():
    list_of_inputs = []

    # Test case 1: 1D integer tensor
    input1 = np.array([1, -3, 5, 0, -2], dtype=np.int32)
    input_dict1 = {"input": input1, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Test case 2: 2D float tensor with dim and keepdim
    input2 = np.array([[1.5, -2.5, 3.5], [4.5, 0.5, -1.5]], dtype=np.float32)
    input_dict2 = {"input": input2, "dim": 0, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Test case 3: 3D tensor without dim
    input3 = np.random.rand(2, 3, 4).astype(np.float64)
    input_dict3 = {"input": input3, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict3))
    
    # Test case 4: 2D int tensor with dim=1
    input4 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    input_dict4 = {"input": input4, "dim": 1, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Test case 5: 1D tensor with NaN
    input5 = np.array([1.0, -3.0, np.nan, 5.0], dtype=np.float32)
    input_dict5 = {"input": input5, "dim": None, "keepdim": False}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Test case 7: 2D tensor with negative values and keepdim=True
    input7 = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32)
    input_dict7 = {"input": input7, "dim": 0, "keepdim": True}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    return list_of_inputs

generated_inputs = aminmax_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('aminmax', generated_inputs)
