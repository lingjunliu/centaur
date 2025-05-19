
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def unsqueeze_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": input1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int tensor, negative dim
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict2 = {"input": input2, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: 3D complex tensor
    input3 = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict3 = {"input": input3, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 4D float tensor, dim in the middle
    input4 = np.random.randn(2, 3, 4, 5).astype(np.float64)
    input_dict4 = {"input": input4, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Scalar tensor
    input5 = np.array(5, dtype=np.int64)
    input_dict5 = {"input": input5, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict5))
    
    return list_of_inputs

generated_inputs = unsqueeze_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('unsqueeze', generated_inputs)
