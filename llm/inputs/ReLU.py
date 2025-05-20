
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def ReLU_inputs():
    list_of_inputs = []

    # Input 1: 1D float tensor with positive and negative values
    input1 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    input_dict1 = {
        "input": input1,
        "inplace": False
    }
    list_of_inputs.append(input_dict1)

    # Input 2: 2D float tensor with only positive values
    input2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict2 = {
        "input": input2,
        "inplace": True
    }
    list_of_inputs.append(input_dict2)

    # Input 3: 3D float tensor with mixed values
    input3 = np.array([[[ -1.0, 2.0], [0.0, -4.0]], [[5.0, -6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict3 = {
        "input": input3,
        "inplace": False
    }
    list_of_inputs.append(input_dict3)

    # Input 4: 1D int tensor with mixed values
    input4 = np.array([-1, 0, 1, 2, -2], dtype=np.int32)
    input_dict4 = {
        "input": input4,
        "inplace": True
    }
    list_of_inputs.append(input_dict4)
    
    # Input 5: 2D int tensor with all negative values
    input5 = np.array([[-1, -2], [-3, -4]], dtype=np.int32)
    input_dict5 = {
        "input": input5,
        "inplace": False
    }
    list_of_inputs.append(input_dict5)


    return list_of_inputs

generated_inputs = ReLU_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('ReLU', generated_inputs)
