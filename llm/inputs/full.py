
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def full_inputs():
    list_of_inputs = []

    input_dict = {
        "size": (2, 3),
        "fill_value": 1.0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (5,),
        "fill_value": 2.5,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (2, 2, 2),
        "fill_value": -1.0,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (1, 4, 4),
        "fill_value": 0.0,
        "dtype": np.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    input_dict = {
        "size": (3, 1),
        "fill_value": 100.0,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = full_inputs()

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        status, exception_message = oracle_crash(api_driver, input_dict, cpu=True)
        if status == "invalid":
            print('Input number' + str(idx+1) + ' got the following exception: ' + exception_message)
            return
    
    print("Valid")

check_valid('full', list_of_inputs)
