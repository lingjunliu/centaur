
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

import torch, copy
import numpy as np

def empty_strided_inputs():
    list_of_inputs = []

    size = (2, 3)
    stride = (3, 1)
    input_dict = {
        "size": size,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    size = (4, 5, 2)
    stride = (10, 2, 1)
    input_dict = {
        "size": size,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    size = (1, 1, 1)
    stride = (1, 1, 1)
    input_dict = {
        "size": size,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    size = (7,)
    stride = (1,)
    input_dict = {
        "size": size,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    size = (2, 2, 2, 2)
    stride = (8, 4, 2, 1)
    input_dict = {
        "size": size,
        "stride": stride
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

list_of_inputs = empty_strided_inputs()

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

check_valid('empty_strided', list_of_inputs)
