
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import copy
import array
import numpy as np

def frombuffer_inputs():
    list_of_inputs = []

    a = array.array('i', [1, 2, 3, 4, 5])
    input_dict = {
        "buffer": a,
        "dtype": torch.int32,
        "count": -1,
        "offset": 0,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = array.array('f', [1.0, 2.0, 3.0, 4.0, 5.0])
    input_dict = {
        "buffer": a,
        "dtype": torch.float32,
        "count": 3,
        "offset": 4,
        "requires_grad": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = array.array('h', [10, 20, 30, 40])
    input_dict = {
        "buffer": a,
        "dtype": torch.int16,
        "count": 2,
        "offset": 0,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = array.array('b', [-1, 0, 0, 0])
    input_dict = {
        "buffer": a,
        "dtype": torch.int32,
        "count": 1,
        "offset": 0,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = array.array('B', [255, 128, 64, 32, 16])
    input_dict = {
        "buffer": a,
        "dtype": torch.uint8,
        "count": -1,
        "offset": 0,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = frombuffer_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('frombuffer', generated_inputs)
