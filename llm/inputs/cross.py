
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy

def cross_inputs():
    list_of_inputs = []

    # Input 1: Basic float tensors
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    other1 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict1 = {"input": input1, "other": other1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Integer tensors
    input2 = np.array([1, 0, 0], dtype=np.int64)
    other2 = np.array([0, 1, 0], dtype=np.int64)
    input_dict2 = {"input": input2, "other": other2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Negative values
    input3 = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    other3 = np.array([4.0, -5.0, 6.0], dtype=np.float32)
    input_dict3 = {"input": input3, "other": other3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Tensors with zeros
    input7 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    other7 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict7 = {"input": input7, "other": other7}
    list_of_inputs.append(copy.deepcopy(input_dict7))
    

    return list_of_inputs

generated_inputs = cross_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('cross', generated_inputs)
