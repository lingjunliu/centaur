
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def unsqueeze_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor
    input_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict_1 = {"input": input_1, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D int tensor
    input_2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict_2 = {"input": input_2, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D complex tensor
    input_3 = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict_3 = {"input": input_3, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 4D bool tensor
    input_4 = np.array([[[[True, False], [False, True]]]], dtype=np.bool_)
    input_dict_4 = {"input": input_4, "dim": 2}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2D float tensor with negative dimension
    input_5 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    input_dict_5 = {"input": input_5, "dim": -1}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 0D integer tensor
    input_6 = np.array(5, dtype=np.int64)
    input_dict_6 = {"input": input_6, "dim": 0}
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: 3D float tensor, dim = 1
    input_7 = np.random.rand(2,3,4).astype(np.float32)
    input_dict_7 = {"input": input_7, "dim": 1}
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    

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
