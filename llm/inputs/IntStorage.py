
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def intstorage_inputs():
    list_of_inputs = []

    input1 = {
        "data": np.array([]).astype(np.int64),
        "size": 0
    }
    list_of_inputs.append(copy.deepcopy(input1))

    input2 = {
        "data": np.array([1]).astype(np.int64),
        "size": 1
    }
    list_of_inputs.append(copy.deepcopy(input2))

    input3 = {
        "data": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.int64),
        "size": 10
    }
    list_of_inputs.append(copy.deepcopy(input3))
    
    input4 = {
        "data": np.arange(100).astype(np.int64),
        "size": 100
    }
    list_of_inputs.append(copy.deepcopy(input4))

    input5 = {
        "data": np.arange(1000).astype(np.int64),
        "size": 1000
    }
    list_of_inputs.append(copy.deepcopy(input5))

    return list_of_inputs

generated_inputs = intstorage_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('IntStorage', generated_inputs)
