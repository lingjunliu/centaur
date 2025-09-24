
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch
import numpy as np
import copy
import array

def frombuffer_inputs():
    list_of_inputs = []

    # Input 1: Basic integer array
    a1 = array.array('i', [1, 2, 3, 4, 5])
    input_dict1 = {
        "buffer": a1,
        "dtype": torch.int32,
        "count": -1,
        "offset": 0,
        "requires_grad": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

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
